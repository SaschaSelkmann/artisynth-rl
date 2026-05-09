"""Sanity-check the variation space the ToyMuscleArmEnv exposes to the agent.

Launches its own ArtiSynth instance (default port 8099 — keep distinct from
training ports) and prints:

  - the declared action space and observation space with semantic labels
  - actual min/max/mean/std per observation dim from N random rollouts,
    plus a flag for dims that hit (clip) the declared bounds
  - reward statistics
  - termination-reason histogram
  - per-episode reference trajectory snippets so you can see how the
    sin/cos targets vary between episodes

Usage:
    cd src/python
    python inspect_env.py --port 8099 --episodes 5 --steps 200 \
        --policy random --save_plot /tmp/toymusclearm_inspection.png
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import artisynth_envs  # noqa: F401  (registers envs)


OBS_LABELS = [
    'joint0_theta',         'joint0_thetaDot',
    'joint1_theta',         'joint1_thetaDot',
    'joint0_ref_theta',     'joint0_ref_thetaDot',
    'joint1_ref_theta',     'joint1_ref_thetaDot',
    'excitation_L0',        'excitation_R0',
    'excitation_L1',        'excitation_R1',
]
OBS_UNITS = [
    'rad', 'rad/s', 'rad', 'rad/s',
    'rad', 'rad/s', 'rad', 'rad/s',
    '-', '-', '-', '-',
]
ACTION_LABELS = ['L0', 'R0', 'L1', 'R1']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='ToyMuscleArmEnv-v0')
    p.add_argument('--port', type=int, default=8099,
                   help='dedicated ArtiSynth port (default 8099 to avoid training)')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--steps', type=int, default=200,
                   help='max steps per episode (env may truncate sooner)')
    p.add_argument('--policy', choices=['random', 'zero', 'half', 'full'],
                   default='random',
                   help='action policy: random uniform, all-0, all-0.5, all-1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gui', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--save_plot', default=None,
                   help='path to write a PNG of per-episode reference trajectories')
    return p.parse_args()


def make_action(space, mode, rng):
    if mode == 'random':
        return rng.uniform(space.low, space.high).astype(np.float32)
    if mode == 'zero':
        return np.zeros(space.shape, dtype=np.float32)
    if mode == 'half':
        return np.full(space.shape, 0.5, dtype=np.float32)
    if mode == 'full':
        return np.ones(space.shape, dtype=np.float32)
    raise ValueError(mode)


def fmt_bounds(low, high, n_dims, labels, units):
    lines = []
    for i in range(n_dims):
        lines.append(f'  [{i:2d}] {labels[i]:24s} ({units[i]:5s}) '
                     f'low={low[i]:+8.3f}  high={high[i]:+8.3f}')
    return '\n'.join(lines)


def fmt_stats(arr, labels, units, low, high):
    """One-line stats per dim; flag dims that touch the declared bounds."""
    lines = []
    head = (f'{"idx":>3s} {"name":24s} {"unit":5s} '
            f'{"min":>9s} {"max":>9s} {"mean":>9s} {"std":>8s} '
            f'{"low%":>6s} {"high%":>6s}  bound?')
    lines.append(head)
    lines.append('-' * len(head))
    for i, name in enumerate(labels):
        col = arr[:, i]
        eps = 1e-3 * max(abs(low[i]), abs(high[i]), 1.0)
        low_pct  = 100.0 * np.mean(col <= low[i]  + eps)
        high_pct = 100.0 * np.mean(col >= high[i] - eps)
        flag = []
        if low_pct  > 0.5: flag.append('LOW')
        if high_pct > 0.5: flag.append('HIGH')
        flag_s = ','.join(flag) or '-'
        lines.append(
            f'{i:3d} {name:24s} {units[i]:5s} '
            f'{col.min():+9.3f} {col.max():+9.3f} {col.mean():+9.3f} {col.std():8.3f} '
            f'{low_pct:6.1f} {high_pct:6.1f}  {flag_s}'
        )
    return '\n'.join(lines)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = gym.make(
        args.env,
        ip='localhost', port=args.port, gui=args.gui,
        seed=args.seed, test=False,
        include_current_state=True, include_current_excitations=True,
        incremental_actions=False, zero_excitations_on_reset=True,
        disable_env_checker=True,
    )
    a_space = env.action_space
    o_space = env.observation_space
    n_obs = o_space.shape[0]
    n_act = a_space.shape[0]

    print(f'\n=== {args.env} — declared spaces ===')
    print(f'action_space:      shape={a_space.shape}  dtype={a_space.dtype}')
    print(fmt_bounds(a_space.low, a_space.high, n_act, ACTION_LABELS, ['-'] * n_act))
    print(f'\nobservation_space: shape={o_space.shape}  dtype={o_space.dtype}')
    if n_obs != len(OBS_LABELS):
        print(f'WARN: env reports {n_obs} obs dims but OBS_LABELS has '
              f'{len(OBS_LABELS)}; truncating labels')
    labels = OBS_LABELS[:n_obs] + [f'extra_{i}' for i in range(len(OBS_LABELS), n_obs)]
    units  = OBS_UNITS[:n_obs]  + ['?']     * max(0, n_obs - len(OBS_UNITS))
    print(fmt_bounds(o_space.low, o_space.high, n_obs, labels, units))

    print(f'\n=== Sampling {args.episodes} episodes × up to {args.steps} steps '
          f'({args.policy} policy) ===')
    all_obs = []
    all_rewards = []
    termination_reasons = Counter()
    per_episode = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_obs = [obs.copy()]
        ep_rewards = []
        for s in range(args.steps):
            a = make_action(a_space, args.policy, rng)
            obs, r, term, trunc, info = env.step(a)
            ep_obs.append(obs.copy())
            ep_rewards.append(r)
            if term:
                termination_reasons[info.get('terminated_reason', 'terminated')] += 1
                break
            if trunc:
                termination_reasons['truncated'] += 1
                break
        ep_obs = np.array(ep_obs)
        all_obs.append(ep_obs)
        all_rewards.extend(ep_rewards)
        # Snapshot of the reference at the end of this episode (gives a feel
        # for how varied the sampled trajectories are).
        rad = ep_obs[-1, 4:8]
        per_episode.append({
            'steps': len(ep_rewards),
            'mean_r': float(np.mean(ep_rewards)) if ep_rewards else float('nan'),
            'min_r':  float(np.min(ep_rewards))  if ep_rewards else float('nan'),
            'final_ref_deg': [math.degrees(rad[0]), math.degrees(rad[2])],
            'final_refdot_rad_s': [float(rad[1]), float(rad[3])],
        })
        print(f'  ep {ep}: {len(ep_rewards):3d} steps, '
              f"r in [{per_episode[-1]['min_r']:+.3f}, "
              f"{max(ep_rewards) if ep_rewards else float('nan'):+.3f}], "
              f"final ref=({per_episode[-1]['final_ref_deg'][0]:+.1f}°, "
              f"{per_episode[-1]['final_ref_deg'][1]:+.1f}°)")

    obs_arr = np.concatenate(all_obs, axis=0)

    print(f'\n=== Observation statistics ({len(obs_arr)} samples) ===')
    print(fmt_stats(obs_arr, labels, units, o_space.low, o_space.high))

    if all_rewards:
        ar = np.array(all_rewards)
        print(f'\n=== Reward (N={len(ar)}) ===')
        print(f'  min={ar.min():+.4f}  max={ar.max():+.4f}  '
              f'mean={ar.mean():+.4f}  std={ar.std():.4f}')
        print(f'  percentiles 5/50/95: '
              f'{np.percentile(ar, 5):+.4f} / '
              f'{np.percentile(ar, 50):+.4f} / '
              f'{np.percentile(ar, 95):+.4f}')

    print('\n=== Termination histogram ===')
    if not termination_reasons:
        print('  (no episode finished within the step budget)')
    for reason, count in termination_reasons.most_common():
        print(f'  {reason:30s} {count:4d}')

    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            for ep, ep_obs in enumerate(all_obs):
                t = np.arange(len(ep_obs))
                axes[0].plot(t, np.degrees(ep_obs[:, 4]), label=f'ep{ep}')
                axes[1].plot(t, np.degrees(ep_obs[:, 6]), label=f'ep{ep}')
            axes[0].set_ylabel('joint0_ref [deg]')
            axes[1].set_ylabel('joint1_ref [deg]')
            axes[1].set_xlabel('step')
            for a in axes:
                a.grid(alpha=0.3)
                a.legend(loc='upper right', fontsize=8)
            fig.suptitle('Reference trajectory variation across episodes')
            fig.tight_layout()
            fig.savefig(args.save_plot, dpi=110)
            print(f'\nplot saved to {args.save_plot}')
        except ImportError:
            print('\nmatplotlib not available — skipping plot')

    env.close()
    print('done')


if __name__ == '__main__':
    main()
