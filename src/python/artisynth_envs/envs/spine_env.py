import numpy as np
from typing import Optional

from common import constants as c
from common.config import setup_logger
from artisynth_envs.artisynth_base_env import ArtiSynthBase

logger = setup_logger()


class SpineEnv(ArtiSynthBase):
    def __init__(self, wait_action, reset_step, goal_reward, goal_threshold,
                 w_u: float = 1.0, w_d: float = 0.1, w_r: float = 0.05, **kwargs):
        super().__init__(**kwargs)

        self.prev_exc = None
        self.episode_counter = 0
        self.reset_step = int(reset_step)
        self.wait_action = float(wait_action)
        self.goal_reward = goal_reward
        self.goal_threshold = goal_threshold
        self.phi_r_episode = []

        self.w_u = w_u
        self.w_d = w_d
        self.w_r = w_r

        self.init_spaces()

    def _calc_reward(self, state, excitations):
        observation = state[c.OBSERVATION_STR]
        h = 1
        w_u = self.w_u ** 2
        w_d = self.w_d
        w_r = self.w_r

        phi_u = self.distance_to_target(observation)
        info = {'distance': phi_u}

        terminated = phi_u < self.goal_threshold
        done_reward = self.goal_reward if terminated else 0.0
        if terminated:
            logger.info('Goal reached: %.4f < %.4f', phi_u, self.goal_threshold)

        phi_d = 0.0
        if self.prev_exc is not None:
            phi_d = np.linalg.norm(excitations - self.prev_exc) / (2 * h)
        self.prev_exc = np.asarray(excitations)

        phi_r = np.linalg.norm(excitations) / 2
        self.phi_r_episode.append(phi_r)

        reward = -(phi_u / (2 * h) * w_u + phi_d * w_d + phi_r * w_r) / 10 + done_reward
        logger.log(19, '%.4f = -(%.4f + %.4f + %.4f)/10', reward, phi_u * w_u, phi_d * w_d, phi_r * w_r)
        return reward, terminated, info

    def step(self, action):
        self.episode_counter += 1
        self.take_action(action)
        if self.wait_action > 0:
            self._sleep_sim(self.wait_action)

        state = self.get_state_dict()
        obs = self.state_dic_to_array(state)
        reward, terminated, info = self._calc_reward(state, action)

        truncated = self.episode_counter >= self.reset_step
        if terminated or truncated:
            info['phi_r_mean'] = float(np.mean(self.phi_r_episode)) if self.phi_r_episode else 0.0

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.prev_exc = None
        self.episode_counter = 0
        self.phi_r_episode = []
        return super().reset(seed=seed, options=options)
