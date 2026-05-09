"""Joint-angle tracking env for the four-muscle ToyMuscleArm.

The agent chooses excitations for the four muscles (L0, R0, L1, R1) and tries
to follow a reference trajectory of joint angles (theta0, theta1) generated
on the Java side. Reference angles + velocities are published as RlProps
(joint0_ref, joint1_ref); current joint state is published as joint0/joint1.

Termination:
  - truncation when step_count >= max_steps  (= episode_duration / wait_action)
  - termination if a joint is pinned at its hard limit for
    `pinned_consecutive_steps` consecutive steps (default 1 s)
  - termination if a joint angular velocity exceeds `velocity_blowup_rad_s`
    for `velocity_blowup_consecutive_steps` consecutive steps
  - termination on NaN observations

These termination criteria are pure physical-failure indicators (arm is
stuck against its hard stop, simulator blew up, NaN). They do **not** look
at tracking error against the reference — that is shaped purely through the
reward, so transient excursions and oscillations around the trajectory are
allowed.

Step counting (rather than sim time) is used because ArtiSynth's headless
scheduler advances simulation time unthrottled — using sim time for episode
boundaries would give variable-length episodes.
"""
import math
from typing import Optional

import numpy as np

from common import constants as c
from common.config import setup_logger
from artisynth_envs.artisynth_base_env import ArtiSynthBase

logger = setup_logger()


# Physical hinge limits of the ToyMuscleArm (rad).
JOINT0_LIMIT_RAD = math.radians(70)
JOINT1_LIMIT_RAD = math.radians(120)
JOINT_LIMITS_RAD = (JOINT0_LIMIT_RAD, JOINT1_LIMIT_RAD)
ANG_VEL_BOUND_RAD_S = 50.0


class ToyMuscleArmEnv(ArtiSynthBase):
    def __init__(self,
                 episode_duration: float = 10.0,
                 wait_action: float = 0.05,
                 # reward shaping
                 goal_threshold_deg: float = 5.0,
                 goal_reward: float = 0.1,
                 w_u: float = 1.0,
                 w_r: float = 0.01,
                 # pinned-at-limit termination
                 pinned_enabled: bool = True,
                 pinned_angle_tol_deg: float = 1.0,
                 pinned_velocity_tol: float = 0.1,
                 pinned_consecutive_steps: int = 20,
                 # velocity-blowup termination
                 velocity_blowup_enabled: bool = True,
                 velocity_blowup_rad_s: float = 30.0,
                 velocity_blowup_consecutive_steps: int = 5,
                 **kwargs):
        super().__init__(**kwargs)

        self.episode_duration = float(episode_duration)
        self.wait_action      = float(wait_action)

        self.goal_threshold = math.radians(float(goal_threshold_deg))
        self.goal_reward    = float(goal_reward)
        self.w_u            = float(w_u)
        self.w_r            = float(w_r)

        self.pinned_enabled              = bool(pinned_enabled)
        self.pinned_angle_tol            = math.radians(float(pinned_angle_tol_deg))
        self.pinned_velocity_tol         = float(pinned_velocity_tol)
        self.pinned_consecutive_steps    = int(pinned_consecutive_steps)

        self.velocity_blowup_enabled              = bool(velocity_blowup_enabled)
        self.velocity_blowup_threshold            = float(velocity_blowup_rad_s)
        self.velocity_blowup_consecutive_steps    = int(velocity_blowup_consecutive_steps)

        # Truncate after this many actions. With wait_action=0.05s and
        # episode_duration=10s the agent makes 200 control decisions per
        # episode regardless of how fast the headless sim advances.
        self.max_steps = max(1, int(round(self.episode_duration / max(self.wait_action, 1e-6))))

        self._step_count    = 0
        self._pinned_count  = [0, 0]   # per joint
        self._blowup_count  = [0, 0]   # per joint

        self.init_spaces()

    # ------------ observation/action spaces ------------

    def get_state_boundaries(self, action_size):
        joint0_lo = np.array([-JOINT0_LIMIT_RAD, -ANG_VEL_BOUND_RAD_S], dtype=np.float32)
        joint0_hi = np.array([+JOINT0_LIMIT_RAD, +ANG_VEL_BOUND_RAD_S], dtype=np.float32)
        joint1_lo = np.array([-JOINT1_LIMIT_RAD, -ANG_VEL_BOUND_RAD_S], dtype=np.float32)
        joint1_hi = np.array([+JOINT1_LIMIT_RAD, +ANG_VEL_BOUND_RAD_S], dtype=np.float32)

        low  = np.concatenate([joint0_lo, joint1_lo,  joint0_lo, joint1_lo])
        high = np.concatenate([joint0_hi, joint1_hi,  joint0_hi, joint1_hi])

        if self.include_current_excitations:
            low  = np.concatenate([low,  np.full(action_size, c.LOW_EXCITATION,  dtype=np.float32)])
            high = np.concatenate([high, np.full(action_size, c.HIGH_EXCITATION, dtype=np.float32)])
        return low.astype(np.float32), high.astype(np.float32)

    def state_dic_to_array(self, js):
        """RlProps for this env live under the `properties` field. Layout:
        [j0_theta, j0_dot, j1_theta, j1_dot,
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
        self._step_count   = 0
        self._pinned_count = [0, 0]
        self._blowup_count = [0, 0]
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._step_count += 1
        # In stepped mode the POST /excitations response carries the
        # post-action state; in free-run mode it's an empty stub.
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
            return obs, -10.0, True, False, {'terminated_reason': 'nan'}

        props      = state['properties']
        theta      = np.array([props['joint0'][0], props['joint1'][0]], dtype=np.float64)
        theta_dot  = np.array([props['joint0'][1], props['joint1'][1]], dtype=np.float64)
        theta_ref  = np.array([props['joint0_ref'][0], props['joint1_ref'][0]], dtype=np.float64)
        err        = np.abs(theta - theta_ref)

        reward, info = self._reward(err, action)
        terminated, term_reason = self._terminated(theta, theta_dot)
        if terminated:
            info['terminated_reason'] = term_reason

        truncated = self._step_count >= self.max_steps
        info['step']     = self._step_count
        info['sim_time'] = float(state.get('time', 0.0))
        info['err0_deg'] = math.degrees(err[0])
        info['err1_deg'] = math.degrees(err[1])

        return obs, reward, terminated, truncated, info

    # ------------ reward & termination ------------

    def _reward(self, err, action):
        phi_u = float(np.dot(err, err))                       # rad^2
        phi_r = float(np.dot(action, action))                 # excitation^2
        max_err = float(np.max(err))
        bonus = self.goal_reward if max_err < self.goal_threshold else 0.0
        reward = -self.w_u * phi_u - self.w_r * phi_r + bonus
        info = {'phi_u': phi_u, 'phi_r': phi_r, 'goal_bonus': bonus}
        return reward, info

    def _terminated(self, theta, theta_dot):
        """Pure physical-failure checks. Tracking error is **not** considered
        here — the reward function shapes that. Returns
        (terminated: bool, reason: str)."""
        if self.pinned_enabled:
            for j in range(2):
                limit       = JOINT_LIMITS_RAD[j]
                at_limit    = abs(abs(theta[j]) - limit) < self.pinned_angle_tol
                near_still  = abs(theta_dot[j]) < self.pinned_velocity_tol
                if at_limit and near_still:
                    self._pinned_count[j] += 1
                    if self._pinned_count[j] >= self.pinned_consecutive_steps:
                        return True, f'pinned_joint{j}'
                else:
                    self._pinned_count[j] = 0

        if self.velocity_blowup_enabled:
            for j in range(2):
                if abs(theta_dot[j]) > self.velocity_blowup_threshold:
                    self._blowup_count[j] += 1
                    if self._blowup_count[j] >= self.velocity_blowup_consecutive_steps:
                        return True, f'velocity_blowup_joint{j}'
                else:
                    self._blowup_count[j] = 0

        return False, ''
