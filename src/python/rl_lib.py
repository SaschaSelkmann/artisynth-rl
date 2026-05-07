import inspect
import os

ALGORITHM_REGISTRY = {
    'SAC': ('stable_baselines3', 'SAC'),
    'TD3': ('stable_baselines3', 'TD3'),
    'PPO': ('stable_baselines3', 'PPO'),
    'TQC': ('sb3_contrib', 'TQC'),
}


def get_algorithm_cls(name: str):
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f'Unknown algorithm "{name}". Available: {list(ALGORITHM_REGISTRY)}'
        )
    module_name, cls_name = ALGORITHM_REGISTRY[name]
    import importlib
    return getattr(importlib.import_module(module_name), cls_name)


def filter_algorithm_kwargs(cls, kwargs: dict) -> dict:
    valid    = set(inspect.signature(cls.__init__).parameters) - {'self', 'policy', 'env'}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    ignored  = set(kwargs) - set(filtered)
    if ignored:
        import logging
        logging.getLogger('artisynth_rl').warning(
            'algorithm_kwargs ignored for %s: %s', cls.__name__, sorted(ignored)
        )
    return filtered


def _base_env_kwargs(config: dict, test: bool = False) -> dict:
    return dict(
        ip=config.get('ip', 'localhost'),
        port=config.get('port', 8080),
        gui=config.get('gui', False),
        seed=config.get('seed', 12345),
        test=test,
        include_current_state=config.get('include_current_state', True),
        include_current_excitations=config.get('include_current_excitations', True),
        incremental_actions=config.get('incremental_actions', False),
        zero_excitations_on_reset=config.get('zero_excitations_on_reset', True),
        goal_threshold=config.get('goal_threshold', 0.1),
        goal_reward=config.get('goal_reward', 5.0),
        reset_step=config.get('reset_step', 200),
        wait_action=config.get('wait_action', 0.0),
    )


def make_env(config: dict, rank: int = 0, test: bool = False):
    import gymnasium as gym
    import artisynth_envs  # noqa: F401

    kwargs = _base_env_kwargs(config, test=test)
    kwargs['port'] += rank
    kwargs['seed'] += rank
    return gym.make(config['env'], **kwargs)


def make_vec_env(config: dict, test: bool = False):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    n_envs = config.get('n_envs', 1)

    def _factory(rank: int):
        def _init():
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            return make_env(config, rank=rank, test=test)
        return _init

    fns = [_factory(i) for i in range(n_envs)]
    return VecMonitor(SubprocVecEnv(fns, start_method='fork'))


def make_model(config: dict, env, run_dir: str):
    algo_name   = config.get('algorithm', 'SAC')
    cls         = get_algorithm_cls(algo_name)
    algo_kwargs = filter_algorithm_kwargs(cls, config.get('algorithm_kwargs', {}))
    tb_log      = os.path.join(run_dir, 'tb')
    return cls(
        'MlpPolicy',
        env,
        verbose=config.get('verbose', 1),
        seed=config.get('seed', 12345),
        tensorboard_log=tb_log,
        **algo_kwargs,
    )


def load_model(config: dict, env, run_dir: str):
    cls        = get_algorithm_cls(config.get('algorithm', 'SAC'))
    model_path = os.path.join(run_dir, 'model')
    tb_log     = os.path.join(run_dir, 'tb')
    return cls.load(model_path, env=env, tensorboard_log=tb_log)


def run_test_episodes(model, env, n_episodes: int = 10) -> list:
    import numpy as np
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f'  Episode {ep + 1:>3}: reward={ep_reward:.3f}')
    print(f'Mean: {np.mean(rewards):.3f}  Std: {np.std(rewards):.3f}')
    return rewards
