"""
Evaluate a trained ArtiSynth RL model.

Usage:
    python test.py --load results/SpineEnv-v0/SAC/20260507_143022/
    python test.py --load results/SpineEnv-v0/SAC/20260507_143022/ --episodes 20 --gui
    python test.py --load results/toymusclearm_baseline/ --episodes 5 --report
    python test.py --load results/toymusclearm_baseline/ --episodes 3 --video
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description='ArtiSynth-RL evaluation')
    p.add_argument('--load',     required=True, help='Run directory (contains model.zip + config.yaml)')
    p.add_argument('--config',   default=None,  help='Override config (optional)')
    p.add_argument('--episodes', type=int,      default=10)
    p.add_argument('--gui',      action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--seed',     type=int,      default=None)
    p.add_argument('--report',   action=argparse.BooleanOptionalAction, default=False,
                   help='Record per-step data and write CSV + PDF report under '
                        '<run_dir>/eval_<timestamp>/')
    p.add_argument('--video',    action=argparse.BooleanOptionalAction, default=False,
                   help='Capture viewer frames and render an MP4/MOV video to '
                        '<run_dir>/eval_<timestamp>/. Implies --gui (the '
                        'recorder needs the OpenGL viewer).')
    p.add_argument('--video-fps', dest='video_fps', type=float, default=20.0,
                   help='Target frame rate for --video. Default 20 = 1 frame '
                        'per stepped-mode RL step (wait_action=0.05) → '
                        'real-time playback.')
    p.add_argument('--episode-duration', dest='episode_duration', type=float, default=None,
                   help='Override the trained episode_duration (seconds). '
                        'Useful for stress-testing whether the policy '
                        'generalises beyond the 10 s window it was trained '
                        'on. wait_action stays as configured so per-step '
                        'dynamics match training.')
    return p.parse_args()


def _resolve_run_dir(load_arg: str) -> str:
    """Map either a run directory or a checkpoint file path back to its run dir."""
    if os.path.isdir(load_arg):
        return load_arg
    parent = os.path.dirname(load_arg)
    return os.path.dirname(parent) if os.path.basename(parent) == 'checkpoints' else parent


def main():
    args = parse_args()

    from config_utils import load_run_config, load_config, merge_cli
    from rl_lib import make_env, load_model, run_test_episodes

    if args.video and not args.gui:
        print('--video requires --gui; enabling GUI mode.')
        args.gui = True

    config = load_run_config(args.load)
    if args.config:
        config.update(load_config(args.config))
    config = merge_cli(config, {
        'gui':              args.gui,
        'seed':             args.seed,
        'episode_duration': args.episode_duration,
    })
    if args.episode_duration is not None:
        print(f'episode_duration override: {args.episode_duration:.1f}s '
              f'(trained: {load_run_config(args.load).get("episode_duration", 10.0)}s)')

    env   = make_env(config, rank=0, test=True)
    model = load_model(config, env, args.load)

    algo     = config.get('algorithm', 'SAC')
    env_id   = config['env']
    run_name = config.get('run_name', os.path.basename(os.path.normpath(args.load)))
    print(f'Evaluating {env_id} ({algo}) for {args.episodes} episodes …')

    run_dir = _resolve_run_dir(args.load)
    eval_dir = None  # only created if --report or --video

    # ----- start video recording before any episode runs ------------------
    if args.video:
        eval_dir = eval_dir or os.path.join(
            run_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(eval_dir, exist_ok=True)
        net = env.unwrapped.net
        try:
            msg = net.send_msg(
                {'name': 'eval', 'outputDir': eval_dir, 'fps': args.video_fps},
                request_type='POST', message='recording/start')
            print(f'Recording: {msg}')
        except Exception as e:
            print(f'WARNING: could not start recording: {e}')
            args.video = False

    # ----- run episodes ---------------------------------------------------
    try:
        if args.report:
            from eval_report import run_and_report
            run_and_report(model, env, n_episodes=args.episodes, run_dir=run_dir,
                           env_id=env_id, run_name=run_name)
        else:
            run_test_episodes(model, env, n_episodes=args.episodes)
    finally:
        # ----- stop video recording always, even on exception -------------
        if args.video:
            try:
                info = env.unwrapped.net.send_msg(
                    {}, request_type='POST', message='recording/stop')
                print(f'Recording stopped: {info}')
                _render_video_with_ffmpeg(info)
            except Exception as e:
                print(f'WARNING: could not stop recording: {e}')

        env.close()


def _render_video_with_ffmpeg(info: dict) -> None:
    """Assemble the PNG frames produced by MovieMaker into an MP4 using the
    system ffmpeg. We do this on the Python side because MovieMaker.render()
    requires the GUI MovieMakerDialog to be open."""
    import shutil
    import subprocess

    frames = int(info.get('frames', 0))
    if frames <= 0:
        print(f'  No frames captured (frames={frames}); skipping ffmpeg.')
        return

    out_dir   = info['outputDir']
    base      = info.get('baseName', 'recording')
    pattern   = info.get('framePattern', 'frame%05d.png')
    fps       = float(info.get('frameRate', 20.0))
    out_file  = os.path.join(out_dir, f'{base}.mp4')

    if shutil.which('ffmpeg') is None:
        print('  ffmpeg not on PATH — leaving PNG frames in place. Install '
              "with 'sudo apt install ffmpeg' and assemble manually:")
        print(f'    cd {out_dir}')
        print(f'    ffmpeg -y -r {fps} -i {pattern} -pix_fmt yuv420p '
              f'-c:v libx264 -crf 20 {base}.mp4')
        return

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'warning',
        '-r', str(fps),
        '-i', os.path.join(out_dir, pattern),
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libx264', '-crf', '20',
        out_file,
    ]
    print(f'  ffmpeg → {out_file}')
    try:
        subprocess.run(cmd, check=True)
        print(f'  Video written: {out_file} ({frames} frames @ {fps} fps)')
    except subprocess.CalledProcessError as e:
        print(f'  ffmpeg failed (exit {e.returncode}); PNG frames are still '
              f'in {out_dir}')


if __name__ == '__main__':
    main()
