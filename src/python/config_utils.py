import os
import yaml
from datetime import datetime


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


def make_run_dir(config: dict) -> str:
    env  = config.get('env', 'unknown')
    algo = config.get('algorithm', 'SAC')
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join('results', env, algo, ts)
    os.makedirs(path, exist_ok=True)
    return path


def merge_cli(config: dict, overrides: dict) -> dict:
    """Apply CLI overrides (non-None values) on top of a config dict."""
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
