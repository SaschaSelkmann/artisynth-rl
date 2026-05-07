"""
Train an ArtiSynth RL environment from a YAML config.

Usage:
    python train.py --config ../../configs/Point2PointEnv-v2.yaml
    python train.py --config ../../configs/SpineEnv-v0.yaml --n_envs 1
    python train.py --config ../../configs/SpineEnv-v0.yaml \\
        --load results/SpineEnv-v0/SAC/20260507_143022/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description='ArtiSynth-RL training')
    p.add_argument('--config',    required=True, help='Path to YAML config file')
    p.add_argument('--load',      default=None,  help='Resume from run directory')
    p.add_argument('--n_envs',    type=int,      default=None)
    p.add_argument('--timesteps', type=int,      default=None)
    p.add_argument('--seed',      type=int,      default=None)
    p.add_argument('--gui',       action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--verbose',   type=int,      default=None)
    return p.parse_args()


def main():
    args = parse_args()

    from config_utils import load_config, save_config, make_run_dir, merge_cli
    from rl_lib import make_vec_env, make_model, load_model, save_model
    from stable_baselines3.common.callbacks import CheckpointCallback

    config = load_config(args.config)
    config = merge_cli(config, {k: v for k, v in vars(args).items()
                                if k not in ('config', 'load')})

    run_dir = make_run_dir(config)
    save_config(config, run_dir)

    n_envs = config.get('n_envs', 1)
    base_port = config.get('port', 8080)
    ports = [base_port + i for i in range(n_envs)]
    print(f'Run directory : {os.path.abspath(run_dir)}')
    print(f'TensorBoard   : tensorboard --logdir {os.path.abspath(os.path.join(run_dir, "tb"))}')
    print(f'Launching {n_envs} environment(s) on port(s) {ports[0]}–{ports[-1]} …')
    if n_envs > 1:
        print('ArtiSynth logs: ' + ', '.join(f'artisynth_{p}.log' for p in ports))

    vec_env = make_vec_env(config)

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    save_freq = max(10_000 // n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix='ckpt',
    )

    if args.load:
        model = load_model(config, vec_env, args.load)
        print(f'Resuming from {args.load}')
    else:
        model = make_model(config, vec_env, run_dir=run_dir)

    algo = config.get('algorithm', 'SAC')
    print(f'Training {algo} on {config["env"]} '
          f'with {n_envs} worker(s) for {config.get("timesteps", 500_000):,} steps …')

    model.learn(
        total_timesteps=config.get('timesteps', 500_000),
        callback=checkpoint_cb,
        progress_bar=True,
        tb_log_name='train',
        log_interval=config.get('log_interval', 10),
    )

    save_model(model, run_dir)
    print(f'Model saved → {os.path.join(run_dir, "model")}.zip')
    vec_env.close()


if __name__ == '__main__':
    main()
