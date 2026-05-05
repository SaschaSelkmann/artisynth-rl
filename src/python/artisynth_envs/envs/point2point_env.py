import numpy as np
from typing import Optional

from common import constants as c
from common.config import setup_logger
from artisynth_envs.artisynth_base_env import ArtiSynthBase

logger = setup_logger()


class Point2PointEnv(ArtiSynthBase):
    def __init__(self, goal_threshold, wait_action, reset_step, goal_reward, **kwargs):
        super().__init__(**kwargs)

        self.goal_threshold = goal_threshold
        self.prev_distance = None
        self.wait_action = wait_action
        self.episode_counter = 0
        self.reset_step = int(reset_step)
        self.goal_reward = goal_reward
        self.position_radius = self._parse_radius(kwargs.get('artisynth_args', ''))
        self.init_spaces()

    @staticmethod
    def _parse_radius(args_str: str) -> float:
        try:
            return float(args_str.split('radius ')[1].split()[0])
        except (IndexError, ValueError):
            return 5.0

    def get_state_boundaries(self, action_size):
        r = self.position_radius
        n_pos_dims = 3
        low  = np.full(n_pos_dims, -r, dtype=np.float32)
        high = np.full(n_pos_dims,  r, dtype=np.float32)

        if self.include_current_state:
            low  = np.concatenate([np.full(n_pos_dims, -r, dtype=np.float32), low])
            high = np.concatenate([np.full(n_pos_dims,  r, dtype=np.float32), high])

        if self.include_current_excitations:
            low  = np.append(low,  np.zeros(action_size,  dtype=np.float32))
            high = np.append(high, np.ones(action_size,   dtype=np.float32))
        return low, high

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.prev_distance = None
        self.episode_counter = 0
        logger.info('Reset')
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.episode_counter += 1
        self.take_action(action)
        if self.wait_action > 0:
            self._sleep_sim(self.wait_action)

        state = self.get_state_dict()
        if not state:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

        obs_dict   = state[c.OBSERVATION_STR]
        distance   = self.distance_to_target(obs_dict)
        reward, terminated, info = self._calculate_reward(distance, self.prev_distance)
        self.prev_distance = distance
        obs = self.state_dic_to_array(state)

        truncated = self.episode_counter >= self.reset_step
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, new_dist, prev_dist):
        if prev_dist is None:
            return 0.0, False, {'distance': new_dist}

        info = {'distance': new_dist}
        if new_dist < self.goal_threshold:
            logger.log(18, 'Goal reached')
            return float(self.goal_reward) if self.goal_reward else 5.0, True, info

        reward = 1.0 / max(self.episode_counter, 1) if prev_dist - new_dist > 0 else -1.0
        logger.log(18, 'Reward: %s', reward)
        return reward, False, info
