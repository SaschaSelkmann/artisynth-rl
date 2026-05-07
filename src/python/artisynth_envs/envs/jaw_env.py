import numpy as np
from typing import Optional

from common import constants as c
from common.config import setup_logger
from artisynth_envs.artisynth_base_env import ArtiSynthBase

logger = setup_logger()


class JawEnv(ArtiSynthBase):
    def __init__(self, wait_action, reset_step, goal_threshold, goal_reward,
                 goal_hack: bool = False,
                 w_u: float = 1.0, w_r: float = 0.3, w_s: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.episode_counter = 0
        self.goal_threshold = float(goal_threshold)
        self.reset_step = int(reset_step)
        self.wait_action = float(wait_action)
        self.goal_reward = goal_reward
        self.goal_hack = goal_hack
        self.goal_th_step = 0

        self.w_u = w_u
        self.w_r = w_r
        self.w_s = w_s

        self.init_spaces(incremental_actions=self.incremental_actions)

    def distance_to_target(self, observation):
        diff = 0.0
        for cur, tgt in zip(self.components[c.CURRENT], self.components[c.TARGET]):
            for prop in self.components[c.PROPS]:
                if prop == 'velocity':
                    continue
                p_cur = np.asarray(observation[cur[c.NAME]][prop])
                p_tgt = np.asarray(observation[tgt[c.NAME]][prop])
                diff += np.linalg.norm(p_cur - p_tgt)
        return diff

    def _get_velocity_distance(self, observation):
        diff = 0.0
        for cur, tgt in zip(self.components[c.CURRENT], self.components[c.TARGET]):
            p_cur = np.asarray(observation[cur[c.NAME]].get('velocity', [0]))
            p_tgt = np.asarray(observation[tgt[c.NAME]].get('velocity', [0]))
            diff += np.linalg.norm(p_cur - p_tgt)
        return diff

    def _non_sym_loss(self, excitations):
        """Penalise asymmetric bilateral muscle activation (pairs placed consecutively)."""
        return np.sum([abs(excitations[i] - excitations[i + 1])
                       for i in range(0, len(excitations) - 1, 2)])

    def _calc_reward(self, state, action):
        observation = state[c.OBSERVATION_STR]
        excitations = np.asarray(state[c.EXCITATIONS_STR])
        muscle_forces = np.asarray(state[c.MUSCLE_FORCES_STR])

        phi_u = self.distance_to_target(observation)
        info = {
            'time':     state.get('time', 0.0),
            'distance': phi_u,
        }

        if self.test_mode:
            info['excitations_each'] = excitations.tolist()
            info['muscleForces_each'] = muscle_forces.tolist()
            info['lowerIncisorPosition'] = observation.get('lowerincisor', {}).get('position')

        reward = -self.w_u * np.log10(phi_u + c.EPSILON)

        terminated = False
        if phi_u < self.goal_threshold:
            if not self.goal_hack:
                terminated = True
            else:
                self.goal_th_step += 1
                if self.goal_th_step >= 5:
                    terminated = True
                    self.goal_th_step = 0
        else:
            self.goal_th_step = 0

        if terminated:
            reward += self.goal_reward
            logger.info('Goal reached: %.4f < %.4f', phi_u, self.goal_threshold)

        phi_r = float(np.linalg.norm(muscle_forces))
        reward -= phi_r * self.w_r

        sym_loss = self._non_sym_loss(excitations)
        reward -= sym_loss * self.w_s

        info['muscleForces'] = phi_r
        info['excitations_mean'] = float(np.mean(excitations))
        info['symmetric_loss'] = float(sym_loss)

        logger.log(18, 'reward=%.4f  dist=%.4f  forces=%.4f', reward, phi_u, phi_r)
        return reward, terminated, info

    def step(self, action):
        action = self.wrap_action(action)
        self.episode_counter += 1
        self.take_action(action)
        if self.wait_action > 0:
            self._sleep_sim(self.wait_action)

        state = self.get_state_dict()
        reward, terminated, info = self._calc_reward(state, action)
        obs = self.state_dic_to_array(state)
        truncated = self.episode_counter >= self.reset_step

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.episode_counter = 0
        self.goal_th_step = 0
        return super().reset(seed=seed, options=options)
