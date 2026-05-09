"""Per-episode data capture + PDF evaluation report for trained ArtiSynth-RL
policies.

Used by `test.py --report`. Records every step's observation, action, reward
and termination flags into memory, writes a CSV and a multi-page PDF report.

The PDF layout is env-aware (a `ToyMuscleArmEnv` page shows joint-angle
tracking; everything else falls back to a generic obs/action layout) so new
envs can plug in by registering a plotter in the registry below.
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Callable, List, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_episodes(model, env, n_episodes: int) -> List[Dict[str, Any]]:
    """Run `n_episodes` deterministic rollouts and capture per-step data."""
    episodes: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        obs, info0 = env.reset()
        steps: List[Dict[str, Any]] = []
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            steps.append({
                'obs':         np.asarray(obs, dtype=np.float64).copy(),
                'action':      np.asarray(action, dtype=np.float64).copy(),
                'reward':      float(reward),
                'terminated':  bool(terminated),
                'truncated':   bool(truncated),
                'info':        dict(info),
            })
            ep_reward += float(reward)
            done = bool(terminated) or bool(truncated)
            obs = next_obs
        episodes.append({
            'reward':         ep_reward,
            'length':         len(steps),
            'steps':          steps,
            'reset_info':     dict(info0),
        })
        print(f'  Episode {ep + 1}: reward={ep_reward:.3f}  len={len(steps)}')
    return episodes


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(episodes: List[Dict[str, Any]], csv_path: str) -> None:
    if not episodes or not episodes[0]['steps']:
        return
    n_obs = len(episodes[0]['steps'][0]['obs'])
    n_act = len(episodes[0]['steps'][0]['action'])
    header = ['episode', 'step', 'reward', 'terminated', 'truncated']
    header += [f'obs{i}' for i in range(n_obs)]
    header += [f'act{i}' for i in range(n_act)]
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for ei, ep in enumerate(episodes):
            for si, s in enumerate(ep['steps']):
                w.writerow([
                    ei, si, s['reward'], int(s['terminated']), int(s['truncated']),
                    *[f'{x:.6f}' for x in s['obs']],
                    *[f'{x:.6f}' for x in s['action']],
                ])


# ---------------------------------------------------------------------------
# PDF — env-specific plotters
# ---------------------------------------------------------------------------

ToyMuscleArmObsLayout = {
    'j0_th':      0,
    'j0_dot':     1,
    'j1_th':      2,
    'j1_dot':     3,
    'j0_ref_th':  4,
    'j0_ref_dot': 5,
    'j1_ref_th':  6,
    'j1_ref_dot': 7,
}
ToyMuscleArmActionNames = ['L0', 'R0', 'L1', 'R1']


def _plot_summary_page(pdf, episodes, env_id, run_name, custom_text=''):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(f'Evaluation report — {env_id}', fontsize=14, weight='bold')

    rewards = np.array([ep['reward'] for ep in episodes])
    lengths = np.array([ep['length'] for ep in episodes])
    text = (
        f'Run:          {run_name}\n'
        f'Generated:    {datetime.now().isoformat(timespec="seconds")}\n'
        f'Episodes:     {len(episodes)}\n'
        f'Reward        mean={rewards.mean():+.3f}  std={rewards.std():.3f}  '
        f'min={rewards.min():+.3f}  max={rewards.max():+.3f}\n'
        f'Length        mean={lengths.mean():.1f}  min={lengths.min()}  max={lengths.max()}\n'
    )
    if custom_text:
        text += '\n' + custom_text

    ax_text = fig.add_axes([0.07, 0.72, 0.86, 0.20])
    ax_text.axis('off')
    ax_text.text(0, 1, text, family='monospace', fontsize=9, va='top')

    ax_rew = fig.add_axes([0.10, 0.40, 0.80, 0.25])
    ax_rew.bar(range(1, len(rewards) + 1), rewards, color='seagreen')
    ax_rew.set_title('Episode rewards')
    ax_rew.set_xlabel('episode')
    ax_rew.set_ylabel('total reward')
    ax_rew.grid(alpha=0.3)

    ax_len = fig.add_axes([0.10, 0.08, 0.80, 0.20])
    ax_len.bar(range(1, len(lengths) + 1), lengths, color='steelblue')
    ax_len.set_title('Episode lengths')
    ax_len.set_xlabel('episode')
    ax_len.set_ylabel('steps')
    ax_len.grid(alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def _episode_page_toymusclearm(pdf, ep_idx, ep):
    import matplotlib.pyplot as plt
    steps = ep['steps']
    if not steps:
        return
    T = np.arange(len(steps))
    obs = np.array([s['obs'] for s in steps])
    act = np.array([s['action'] for s in steps])
    r   = np.array([s['reward'] for s in steps])

    j0     = np.degrees(obs[:, ToyMuscleArmObsLayout['j0_th']])
    j0_ref = np.degrees(obs[:, ToyMuscleArmObsLayout['j0_ref_th']])
    j1     = np.degrees(obs[:, ToyMuscleArmObsLayout['j1_th']])
    j1_ref = np.degrees(obs[:, ToyMuscleArmObsLayout['j1_ref_th']])
    err0   = j0 - j0_ref
    err1   = j1 - j1_ref

    fig, axes = plt.subplots(5, 1, figsize=(8.5, 11), sharex=True)
    axes[0].plot(T, j0_ref, '--', color='gray',     lw=1.0, label='target')
    axes[0].plot(T, j0,     '-',  color='steelblue', lw=1.4, label='actual')
    axes[0].set_ylabel('joint0 [deg]')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(T, j1_ref, '--', color='gray',       lw=1.0, label='target')
    axes[1].plot(T, j1,     '-',  color='darkorange', lw=1.4, label='actual')
    axes[1].set_ylabel('joint1 [deg]')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(T, err0, color='steelblue',  lw=1.0, label='err joint0')
    axes[2].plot(T, err1, color='darkorange', lw=1.0, label='err joint1')
    axes[2].axhline(0, color='gray', lw=0.6)
    axes[2].set_ylabel('error [deg]')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(alpha=0.3)

    for i, name in enumerate(ToyMuscleArmActionNames):
        axes[3].plot(T, act[:, i], lw=1.0, label=name)
    axes[3].set_ylabel('excitation')
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc='upper right', ncol=4, fontsize=8)
    axes[3].grid(alpha=0.3)

    axes[4].plot(T, r, color='crimson', lw=1.0)
    axes[4].set_ylabel('reward')
    axes[4].set_xlabel('step')
    axes[4].grid(alpha=0.3)

    rmse0 = float(np.sqrt(np.mean(err0 ** 2)))
    rmse1 = float(np.sqrt(np.mean(err1 ** 2)))
    fig.suptitle(
        f'Episode {ep_idx + 1} — reward {ep["reward"]:+.3f}, length {ep["length"]}, '
        f'RMSE: j0 {rmse0:.2f}°  j1 {rmse1:.2f}°',
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def _toymusclearm_summary_text(episodes):
    err0 = []
    err1 = []
    for ep in episodes:
        for s in ep['steps']:
            err0.append(math.degrees(
                s['obs'][ToyMuscleArmObsLayout['j0_th']]
                - s['obs'][ToyMuscleArmObsLayout['j0_ref_th']]))
            err1.append(math.degrees(
                s['obs'][ToyMuscleArmObsLayout['j1_th']]
                - s['obs'][ToyMuscleArmObsLayout['j1_ref_th']]))
    err0 = np.array(err0)
    err1 = np.array(err1)
    return (
        'Joint-angle tracking error [deg] (over all steps):\n'
        f'  Joint 0  RMSE {np.sqrt(np.mean(err0**2)):.2f}  '
        f'P95 {np.percentile(np.abs(err0), 95):.2f}  '
        f'max {np.max(np.abs(err0)):.2f}\n'
        f'  Joint 1  RMSE {np.sqrt(np.mean(err1**2)):.2f}  '
        f'P95 {np.percentile(np.abs(err1), 95):.2f}  '
        f'max {np.max(np.abs(err1)):.2f}'
    )


def _episode_page_generic(pdf, ep_idx, ep):
    import matplotlib.pyplot as plt
    steps = ep['steps']
    if not steps:
        return
    T = np.arange(len(steps))
    obs = np.array([s['obs'] for s in steps])
    act = np.array([s['action'] for s in steps])
    r   = np.array([s['reward'] for s in steps])

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    for i in range(obs.shape[1]):
        axes[0].plot(T, obs[:, i], lw=0.8, label=f'obs[{i}]')
    axes[0].set_ylabel('observation')
    axes[0].legend(loc='upper right', fontsize=7, ncol=4)
    axes[0].grid(alpha=0.3)

    for i in range(act.shape[1]):
        axes[1].plot(T, act[:, i], lw=1.0, label=f'a[{i}]')
    axes[1].set_ylabel('action')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(T, r, color='crimson', lw=1.0)
    axes[2].set_ylabel('reward')
    axes[2].set_xlabel('step')
    axes[2].grid(alpha=0.3)

    fig.suptitle(f'Episode {ep_idx + 1} — reward {ep["reward"]:+.3f}, '
                 f'length {ep["length"]}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


# Registry: env-id substring → (per-episode plotter, summary-text builder)
PLOTTERS: Dict[str, Dict[str, Callable]] = {
    'ToyMuscleArm': {
        'episode_page': _episode_page_toymusclearm,
        'summary_text': _toymusclearm_summary_text,
    },
}


def make_pdf(episodes, env_id: str, output_path: str, run_name: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plotter = None
    for key, fns in PLOTTERS.items():
        if key in env_id:
            plotter = fns
            break

    custom = plotter['summary_text'](episodes) if plotter else ''
    episode_page = plotter['episode_page'] if plotter else _episode_page_generic

    with PdfPages(output_path) as pdf:
        _plot_summary_page(pdf, episodes, env_id, run_name, custom_text=custom)
        for ei, ep in enumerate(episodes):
            episode_page(pdf, ei, ep)


# ---------------------------------------------------------------------------
# High-level entry point used by test.py
# ---------------------------------------------------------------------------

def run_and_report(model, env, n_episodes: int, run_dir: str,
                   env_id: str, run_name: str) -> str:
    """Record episodes, write CSV + PDF under
    `<run_dir>/eval_<timestamp>/`. Returns the PDF path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(run_dir, f'eval_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Recording {n_episodes} episodes to {out_dir} …')
    episodes = record_episodes(model, env, n_episodes)

    csv_path = os.path.join(out_dir, 'data.csv')
    save_csv(episodes, csv_path)
    print(f'CSV  → {csv_path}')

    pdf_path = os.path.join(out_dir, 'report.pdf')
    make_pdf(episodes, env_id, pdf_path, run_name)
    print(f'PDF  → {pdf_path}')
    return pdf_path
