"""
Hyperparameter optimisation with Optuna.

Usage:
    python optimize.py --config ../../configs/Point2PointEnv-v2.yaml
    python optimize.py --config ../../configs/SpineEnv-v0.yaml --n_trials 100 --n_jobs 2

The search space is defined in the YAML config under the 'optuna' key:

    optuna:
      n_trials: 50
      trial_timesteps: 100000   # steps per trial (default: timesteps // 5)
      eval_episodes: 5
      search_space:
        learning_rate: {type: float, low: 1.0e-5, high: 1.0e-3, log: true}
        batch_size:    {type: categorical, choices: [64, 128, 256, 512]}
        gamma:         {type: float, low: 0.9, high: 0.9999}
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description='ArtiSynth-RL hyperparameter optimisation (Optuna)')
    p.add_argument('--config',     required=True)
    p.add_argument('--n_trials',   type=int, default=None,
                   help='Override optuna.n_trials from config')
    p.add_argument('--n_jobs',     type=int, default=1,
                   help='Parallel Optuna workers (each runs its own vec_env)')
    p.add_argument('--study_name', default=None)
    p.add_argument('--storage',    default=None,
                   help='Optuna storage URL (default: SQLite in results/)')
    return p.parse_args()


def _sample_params(trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        t = spec['type']
        if t == 'float':
            params[name] = trial.suggest_float(
                name, spec['low'], spec['high'], log=spec.get('log', False)
            )
        elif t == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'])
        elif t == 'categorical':
            params[name] = trial.suggest_categorical(name, spec['choices'])
    return params


def _make_objective(base_config: dict):
    from config_utils import make_run_dir, save_config
    from rl_lib import make_vec_env, make_model, make_env, run_test_episodes
    import numpy as np

    optuna_cfg   = base_config.get('optuna', {})
    search_space = optuna_cfg.get('search_space', {})
    trial_steps  = optuna_cfg.get('trial_timesteps',
                                  base_config.get('timesteps', 500_000) // 5)
    n_eval       = optuna_cfg.get('eval_episodes', 5)

    def objective(trial):
        config = dict(base_config)
        sampled = _sample_params(trial, search_space)
        algo_kwargs = dict(config.get('algorithm_kwargs', {}))
        algo_kwargs.update(sampled)
        config['algorithm_kwargs'] = algo_kwargs

        run_dir = make_run_dir(config)
        save_config(config, run_dir)

        vec_env  = make_vec_env(config)
        model    = make_model(config, vec_env, run_dir=run_dir)
        model.learn(
            total_timesteps=trial_steps,
            progress_bar=False,
            log_interval=base_config.get('log_interval', 10),
        )

        eval_env = make_env(config, rank=0, test=True)
        rewards  = run_test_episodes(model, eval_env, n_episodes=n_eval)
        eval_env.close()
        vec_env.close()

        return float(np.mean(rewards))

    return objective


def main():
    args = parse_args()

    try:
        import optuna
    except ImportError:
        raise ImportError('Install optuna first: pip install optuna')

    from config_utils import load_config, save_config, get_run_dir

    config     = load_config(args.config)
    optuna_cfg = config.get('optuna', {})
    n_trials   = args.n_trials or optuna_cfg.get('n_trials', 50)
    run_dir    = get_run_dir(config)
    study_name = args.study_name or config.get('run_name', f'{config["env"]}_{config.get("algorithm","SAC")}')

    if args.storage:
        storage = args.storage
    else:
        db_dir  = os.path.join(run_dir, 'optuna')
        os.makedirs(db_dir, exist_ok=True)
        storage = f'sqlite:///{os.path.join(db_dir, "study.db")}'

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True,
    )

    print(f'Optuna study "{study_name}" — {n_trials} trial(s), {args.n_jobs} job(s)')
    print(f'Storage: {storage}')
    study.optimize(_make_objective(config), n_trials=n_trials, n_jobs=args.n_jobs)

    print(f'\nBest trial : #{study.best_trial.number}')
    print(f'Best value : {study.best_value:.3f}')
    print(f'Best params: {study.best_params}')

    best_config = dict(config)
    best_config['algorithm_kwargs'] = dict(config.get('algorithm_kwargs', {}))
    best_config['algorithm_kwargs'].update(study.best_params)
    best_dir = os.path.join(run_dir, 'optuna', 'best')
    os.makedirs(best_dir, exist_ok=True)
    save_config(best_config, best_dir)
    print(f'Best config → {best_dir}/config.yaml')


if __name__ == '__main__':
    main()
