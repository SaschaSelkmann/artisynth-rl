"""
Train an ArtiSynth RL environment with Stable-Baselines3 SAC using multiple
parallel environments (SubprocVecEnv). Each worker gets its own ArtiSynth
instance on a separate port (base_port, base_port+1, ...).

Usage:
    python main_sb3_parallel.py --env Point2PointEnv-v2 --n_envs 4
    python main_sb3_parallel.py --env Point2PointEnv-v2 --n_envs 4 \
        --load results/Point2PointEnv-v2/sac --test --test_episodes 20
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description='ArtiSynth-RL parallel training via Stable-Baselines3 SAC')

    # Environment
    p.add_argument('--env',    default='Point2PointEnv-v2')
    p.add_argument('--ip',     default='localhost')
    p.add_argument('--port',   type=int, default=8080,
                   help='Base port; worker i connects to port+i')
    p.add_argument('--n_envs', type=int, default=4,
                   help='Number of parallel environments (ArtiSynth instances)')
    p.add_argument('--gui',    action='store_true', default=False)
    p.add_argument('--seed',   type=int, default=12345)

    # Environment kwargs (forwarded to ArtiSynthBase)
    p.add_argument('--include_current_state',       action='store_true', default=True)
    p.add_argument('--include_current_excitations', action='store_true', default=True)
    p.add_argument('--incremental_actions',         action='store_true', default=False)
    p.add_argument('--zero_excitations_on_reset',   action='store_true', default=True)
    p.add_argument('--goal_threshold', type=float, default=0.1)
    p.add_argument('--goal_reward',    type=float, default=5.0)
    p.add_argument('--reset_step',     type=int,   default=200)
    p.add_argument('--wait_action',    type=float, default=0.0)
    p.add_argument('--w_u', type=float, default=1.0)
    p.add_argument('--w_d', type=float, default=0.0)
    p.add_argument('--w_s', type=float, default=0.0)
    p.add_argument('--w_r', type=float, default=0.0)

    # SAC hyper-parameters
    p.add_argument('--timesteps',       type=int,   default=500_000)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--buffer_size',     type=int,   default=100_000)
    p.add_argument('--learning_starts', type=int,   default=1000)
    p.add_argument('--tau',             type=float, default=0.005)
    p.add_argument('--gamma',           type=float, default=0.99)
    p.add_argument('--ent_coef',        default='auto')

    # I/O
    p.add_argument('--save_path', default=None)
    p.add_argument('--load',      default=None)
    p.add_argument('--test',      action='store_true', default=False)
    p.add_argument('--test_episodes', type=int, default=10)
    p.add_argument('--verbose',   type=int, default=1)
    return p.parse_args()


def _env_factory(args, rank):
    """Return a callable that creates one environment for the given worker rank."""
    def _init():
        # Each subprocess needs sys.path set up independently.
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        import gymnasium as gym
        import artisynth_envs  # noqa: F401

        env = gym.make(
            args.env,
            ip=args.ip,
            port=args.port + rank,
            gui=args.gui,
            seed=args.seed + rank,
            test=args.test,
            include_current_state=args.include_current_state,
            include_current_excitations=args.include_current_excitations,
            incremental_actions=args.incremental_actions,
            zero_excitations_on_reset=args.zero_excitations_on_reset,
            goal_threshold=args.goal_threshold,
            goal_reward=args.goal_reward,
            reset_step=args.reset_step,
            wait_action=args.wait_action,
            w_u=args.w_u, w_d=args.w_d, w_s=args.w_s, w_r=args.w_r,
        )
        return env
    return _init


def make_vec_env(args):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    fns = [_env_factory(args, rank) for rank in range(args.n_envs)]
    vec_env = SubprocVecEnv(fns, start_method='fork')
    vec_env = VecMonitor(vec_env)
    return vec_env


def make_single_env(args):
    """Single env used for evaluation."""
    import gymnasium as gym
    import artisynth_envs  # noqa: F401

    return gym.make(
        args.env,
        ip=args.ip,
        port=args.port,
        gui=args.gui,
        seed=args.seed,
        test=args.test,
        include_current_state=args.include_current_state,
        include_current_excitations=args.include_current_excitations,
        incremental_actions=args.incremental_actions,
        zero_excitations_on_reset=args.zero_excitations_on_reset,
        goal_threshold=args.goal_threshold,
        goal_reward=args.goal_reward,
        reset_step=args.reset_step,
        wait_action=args.wait_action,
        w_u=args.w_u, w_d=args.w_d, w_s=args.w_s, w_r=args.w_r,
    )


def main():
    args = parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    save_path = args.save_path or os.path.join('results', args.env, 'sac_parallel')
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    if args.test:
        if args.load is None:
            raise ValueError('--load is required in test mode')
        env = make_single_env(args)
        model = SAC.load(args.load, env=env)
        print(f'Loaded model from {args.load}')
        rewards = []
        for ep in range(args.test_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
            print(f'  Episode {ep+1}: reward={ep_reward:.3f}')
        print(f'Mean reward over {args.test_episodes} episodes: {np.mean(rewards):.3f}')
        env.close()
        return

    print(f'Launching {args.n_envs} parallel environments on ports '
          f'{args.port}–{args.port + args.n_envs - 1} …')
    vec_env = make_vec_env(args)

    ent_coef = args.ent_coef
    try:
        ent_coef = float(ent_coef)
    except ValueError:
        pass

    if args.load is not None:
        model = SAC.load(args.load, env=vec_env)
        print(f'Resuming training from {args.load}')
    else:
        model = SAC(
            'MlpPolicy',
            vec_env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=ent_coef,
            verbose=args.verbose,
            seed=args.seed,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=os.path.dirname(save_path) or '.',
        name_prefix='sac_checkpoint',
    )

    print(f'Training SAC on {args.env} with {args.n_envs} workers '
          f'for {args.timesteps:,} steps …')
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True)

    model.save(save_path)
    print(f'Model saved to {save_path}')
    vec_env.close()


if __name__ == '__main__':
    main()
