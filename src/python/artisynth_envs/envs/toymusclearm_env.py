"""Joint-angle tracking env for the four-muscle ToyMuscleArm.

The agent chooses excitations for the four muscles (L0, R0, L1, R1) and tries
to follow a reference trajectory of joint angles (theta0, theta1) generated
on the Java side. Reference angles + velocities are published as RlProps
(joint0_ref, joint1_ref); current joint state is published as joint0/joint1.

Termination rules:
  - truncation when step_count >= max_steps  (≈ episode_duration / wait_action)
  - termination if any joint error exceeds max_abs_error
  - termination if any joint error rises monotonically for divergence_window steps
  - termination on NaN observations

Step counting (rather than sim time) is used because ArtiSynth's headless
scheduler advances simulation time unthrottled — `info['time']` jumps by
seconds per Python step and would truncate every episode after one action.
"""
import math
from collections import deque
from typing import Optional

import numpy as np

from common import constants as c
from common.config import setup_logger
from artisynth_envs.artisynth_base_env import ArtiSynthBase

logger = setup_logger()


# Physical hinge limits of the ToyMuscleArm (rad). Velocity bound is generous —
# muscle-driven kicks easily reach 30 rad/s during exploration before the agent
# has learned to coordinate the antagonists.
JOINT0_LIMIT_RAD = math.radians(70)
JOINT1_LIMIT_RAD = math.radians(120)
ANG_VEL_BOUND_RAD_S = 50.0


class ToyMuscleArmEnv(ArtiSynthBase):
    def __init__(self,
                 episode_duration: float = 10.0,
                 wait_action: float = 0.05,
                 max_abs_error_deg: float = 60.0,
                 goal_threshold_deg: float = 5.0,
                 goal_reward: float = 0.1,
                 divergence_window: int = 10,
                 w_u: float = 1.0,
                 w_r: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)

        self.episode_duration = float(episode_duration)
        self.wait_action = float(wait_action)
        self.max_abs_error = math.radians(float(max_abs_error_deg))
        self.goal_threshold = math.radians(float(goal_threshold_deg))
        self.goal_reward = float(goal_reward)
        self.divergence_window = int(divergence_window)
        self.w_u = float(w_u)
        self.w_r = float(w_r)

        # Truncate after this many actions. With wait_action=0.05s and
        # episode_duration=10s the agent makes 200 control decisions per
        # episode regardless of how fast the headless sim advances.
        self.max_steps = max(1, int(round(self.episode_duration / max(self.wait_action, 1e-6))))

        self._err_history = [deque(maxlen=self.divergence_window) for _ in range(2)]
        self._step_count = 0

        self.init_spaces()

    # ------------ observation/action spaces ------------

    def get_state_boundaries(self, action_size):
        # joint0 = [theta, thetaDot], joint1 = [theta, thetaDot]
        joint_lo = np.array([-JOINT0_LIMIT_RAD, -ANG_VEL_BOUND_RAD_S], dtype=np.float32)
        joint_hi = np.array([+JOINT0_LIMIT_RAD, +ANG_VEL_BOUND_RAD_S], dtype=np.float32)
        joint1_lo = np.array([-JOINT1_LIMIT_RAD, -ANG_VEL_BOUND_RAD_S], dtype=np.float32)
        joint1_hi = np.array([+JOINT1_LIMIT_RAD, +ANG_VEL_BOUND_RAD_S], dtype=np.float32)

        low = np.concatenate([joint_lo, joint1_lo])    # current
        high = np.concatenate([joint_hi, joint1_hi])
        low = np.concatenate([low, joint_lo, joint1_lo])  # reference
        high = np.concatenate([high, joint_hi, joint1_hi])

        if self.include_current_excitations:
            low = np.concatenate([low,
                                  np.full(action_size, c.LOW_EXCITATION,  dtype=np.float32)])
            high = np.concatenate([high,
                                   np.full(action_size, c.HIGH_EXCITATION, dtype=np.float32)])
        return low.astype(np.float32), high.astype(np.float32)

    def state_dic_to_array(self, js):
        """RlProps for this env live under the `properties` field, not
        `observation`. Layout: [j0_theta, j0_dot, j1_theta, j1_dot,
        j0_ref_theta, j0_ref_dot, j1_ref_theta, j1_ref_dot, *excitations]."""
        props = js.get('properties', {}) or {}
        try:
            j0     = props['joint0']
            j1     = props['joint1']
            j0_ref = props['joint0_ref']
            j1_ref = props['joint1_ref']
        except KeyError as e:
            logger.error('Missing RlProp %s in state response: %s', e, list(props.keys()))
            raise

        obs = np.array(j0 + j1 + j0_ref + j1_ref, dtype=np.float32)
        if self.include_current_excitations:
            obs = np.append(obs, js.get(c.EXCITATIONS_STR, []))
        return obs.astype(np.float32)

    # ------------ episode lifecycle ------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        for d in self._err_history:
            d.clear()
        self._step_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._step_count += 1
        # In stepped mode the POST /excitations response carries the
        # post-action state. In legacy/free-run mode it's an empty stub —
        # then we fall back to a separate GET and (optionally) wait.
        state = self.take_action(action)
        if not state or not state.get('properties'):
            if self.wait_action > 0:
                self._sleep_sim(self.wait_action)
            state = self.get_state_dict()

        if not state:
            return (np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0, True, False, {'error': 'empty state'})

        obs = self.state_dic_to_array(state)
        if not np.all(np.isfinite(obs)):
            logger.warning('Non-finite observation; terminating episode.')
            return obs, -10.0, True, False, {'error': 'nan'}

        # Per-joint absolute errors (rad).
        err = np.abs(np.array([
            state['properties']['joint0'][0] - state['properties']['joint0_ref'][0],
            state['properties']['joint1'][0] - state['properties']['joint1_ref'][0],
        ], dtype=np.float64))

        reward, info = self._reward(err, action)
        terminated = self._terminated(err, info)
        truncated = self._step_count >= self.max_steps
        info['step'] = self._step_count
        info['sim_time'] = float(state.get('time', 0.0))
        info['err0_deg'] = math.degrees(err[0])
        info['err1_deg'] = math.degrees(err[1])

        return obs, reward, terminated, truncated, info

    # ------------ reward & termination ------------

    def _reward(self, err, action):
        phi_u = float(np.dot(err, err))                       # rad^2
        phi_r = float(np.dot(action, action))                 # action^2
        max_err = float(np.max(err))
        bonus = self.goal_reward if max_err < self.goal_threshold else 0.0
        reward = -self.w_u * phi_u - self.w_r * phi_r + bonus
        info = {'phi_u': phi_u, 'phi_r': phi_r, 'goal_bonus': bonus}
        return reward, info

    def _terminated(self, err, info):
        if float(np.max(err)) > self.max_abs_error:
            info['terminated_reason'] = 'max_abs_error'
            return True

        for i in range(2):
            self._err_history[i].append(float(err[i]))

        for i, hist in enumerate(self._err_history):
            if (len(hist) == self.divergence_window
                    and all(b > a for a, b in zip(hist, list(hist)[1:]))):
                info['terminated_reason'] = f'divergence_joint{i}'
                return True
        return False
