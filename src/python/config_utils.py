import os
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_run_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No config.yaml found in {run_dir}')
    return load_config(path)


def save_config(config: dict, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_run_dir(config: dict) -> str:
    run_name = config.get('run_name') or \
        f'{config.get("env", "run")}_{config.get("algorithm", "SAC")}'
    return os.path.join('results', run_name)


def make_run_dir(config: dict) -> str:
    path = get_run_dir(config)
    os.makedirs(path, exist_ok=True)
    return path


def merge_cli(config: dict, overrides: dict) -> dict:
    """Apply CLI overrides (non-None values) on top of a config dict."""
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
