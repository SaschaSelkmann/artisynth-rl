"""
Evaluate a trained ArtiSynth RL model.

Usage:
    python test.py --load results/SpineEnv-v0/SAC/20260507_143022/
    python test.py --load results/SpineEnv-v0/SAC/20260507_143022/ --episodes 20 --gui
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description='ArtiSynth-RL evaluation')
    p.add_argument('--load',     required=True, help='Run directory (contains model.zip + config.yaml)')
    p.add_argument('--config',   default=None,  help='Override config (optional)')
    p.add_argument('--episodes', type=int,      default=10)
    p.add_argument('--gui',      action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--seed',     type=int,      default=None)
    return p.parse_args()


def main():
    args = parse_args()

    from config_utils import load_run_config, load_config, merge_cli
    from rl_lib import make_env, load_model, run_test_episodes

    config = load_run_config(args.load)
    if args.config:
        config.update(load_config(args.config))
    config = merge_cli(config, {'gui': args.gui, 'seed': args.seed})

    env = make_env(config, rank=0, test=True)
    model = load_model(config, env, args.load)

    algo = config.get('algorithm', 'SAC')
    print(f'Evaluating {config["env"]} ({algo}) for {args.episodes} episodes …')
    run_test_episodes(model, env, n_episodes=args.episodes)
    env.close()


if __name__ == '__main__':
    main()
