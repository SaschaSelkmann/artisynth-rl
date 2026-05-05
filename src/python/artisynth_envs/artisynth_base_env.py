import logging
import os
import time as _time
import numpy as np
import subprocess
from abc import ABC
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from common.rest_client import RestClient
import common.constants as c

logger = logging.getLogger(c.LOGGER_STR)


class ArtiSynthBase(gym.Env, ABC):
    metadata = {"render_modes": []}

    def __init__(self, ip, port, artisynth_model, test, components, zero_excitations_on_reset,
                 include_current_excitations, include_current_state, w_s, w_u, w_d, w_r, seed,
                 incremental_actions, gui, artisynth_args='', **kwargs):
        super().__init__()
        if kwargs:
            logger.warning(f'Unused kwargs: {list(kwargs.keys())}')

        self.ip = ip
        self.port = port
        self.test_mode = test

        self.include_current_excitations = include_current_excitations
        self.include_current_state = include_current_state
        self.incremental_actions = incremental_actions

        self.w_u = w_u
        self.w_d = w_d
        self.w_r = w_r
        self.w_s = w_s

        self.action_size = 0
        self.obs_size = 0
        self.components = components
        self.zero_excitations_on_reset = zero_excitations_on_reset

        self.net = RestClient(ip, port)
        if not RestClient.server_is_alive(ip, port):
            self.run_artisynth(ip, port, artisynth_model, gui, artisynth_args)

        self.set_test_mode(test)
        if seed is not None:
            self.net.send_msg(seed, request_type=c.POST_STR, message=c.SET_SEED_STR)

    def set_test_mode(self, test_mode):
        self.net.send_msg(test_mode, request_type=c.POST_STR, message=c.SET_TEST_STR)

    def init_spaces(self, incremental_actions=False):
        info = self.net.send_msg(request_type=c.GET_STR, message=c.INFO_STR)
        action_size = info.get('actionSize', self.get_action_size())
        obs_size    = info.get('obsSize',    self.get_obs_size())

        obs, _ = self.reset()
        state_size = obs.shape[0]

        state_low, state_high = self.get_state_boundaries(action_size)
        assert state_size == state_low.shape[0], (
            f'state_low ({state_low.shape[0]}) vs obs ({state_size}) mismatch')

        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        low_a  = c.LOW_EXCITATION_INC  if incremental_actions else c.LOW_EXCITATION
        high_a = c.HIGH_EXCITATION_INC if incremental_actions else c.HIGH_EXCITATION
        self.action_space = spaces.Box(low=low_a, high=high_a,
                                       shape=(action_size,), dtype=np.float32)

        logger.info('obs_size=%d  action_size=%d  state_size=%d', obs_size, action_size, state_size)
        return action_size, obs_size

    def get_state_boundaries(self, action_size):
        low, high = [], []
        if self.include_current_state:
            for obj in self.components[c.CURRENT]:
                low.extend(obj[c.LOW])
                high.extend(obj[c.HIGH])
        for obj in self.components[c.TARGET]:
            low.extend(obj[c.LOW])
            high.extend(obj[c.HIGH])
        low  = np.array(low,  dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        if self.include_current_excitations:
            low  = np.append(low,  np.full(action_size, c.LOW_EXCITATION,  dtype=np.float32))
            high = np.append(high, np.full(action_size, c.HIGH_EXCITATION, dtype=np.float32))
        return low, high

    def run_artisynth(self, ip, port, artisynth_model, gui, artisynth_args=''):
        if ip not in ('localhost', '0.0.0.0', '127.0.0.1'):
            raise NotImplementedError('Cannot launch ArtiSynth on a remote host.')
        if RestClient.server_is_alive(ip, port):
            return
        cmd = (f'artisynth -model {artisynth_model} '
               f'[ -port {port} {artisynth_args} ] -play -noTimeline')
        if not gui:
            cmd += ' -noGui'
        with open(os.devnull, 'w') as devnull:
            subprocess.Popen(cmd.split(), stdout=devnull, stderr=devnull)
        while not RestClient.server_is_alive(ip, port):
            logger.info('Waiting for ArtiSynth at port %d …', port)
            time.sleep(3)

    # --- low-level REST helpers ---
    def get_obs_size(self):
        return self.net.send_msg(request_type=c.GET_STR, message=c.OBS_SIZE_STR)

    def get_state_size(self):
        return self.net.send_msg(request_type=c.GET_STR, message=c.STATE_SIZE_STR)

    def get_action_size(self):
        return self.net.send_msg(request_type=c.GET_STR, message=c.ACTION_SIZE_STR)

    def get_state_dict(self):
        return self.net.send_msg(request_type=c.GET_STR, message=c.STATE_STR)

    def get_excitations_dict(self):
        return self.net.send_msg(request_type=c.GET_STR, message=c.EXCITATIONS_STR)

    # --- Gymnasium API ---

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.net.send_msg(seed, request_type=c.POST_STR, message=c.SET_SEED_STR)

        state_dict = self.net.send_msg(
            self.zero_excitations_on_reset,
            request_type=c.POST_STR,
            message=c.RESET_STR,
        )
        if not state_dict:
            state_dict = self.get_state_dict()

        if not self.test_mode:
            _time.sleep(0.05)  # give the simulation thread time to settle

        obs = self.state_dic_to_array(state_dict)
        info = {'time': state_dict.get('time', 0.0)}
        return obs, info

    def step(self, action):
        raise NotImplementedError

    def take_action(self, action):
        action = np.clip(action, c.LOW_EXCITATION, c.HIGH_EXCITATION)
        logger.debug('excitations: %s', action)
        next_state_dict = self.net.send_msg(
            {c.EXCITATIONS_STR: action.tolist()},
            request_type=c.POST_STR,
            message=c.EXCITATIONS_STR,
        )
        return next_state_dict

    def state_dic_to_array(self, js):
        observation = js.get(c.OBSERVATION_STR, {})
        obs_vec = np.array([], dtype=np.float32)

        if self.include_current_state:
            for comp in self.components[c.CURRENT]:
                t = observation[comp[c.NAME]]
                for prop in self.components[c.PROPS]:
                    obs_vec = np.append(obs_vec, t[prop])

        for comp in self.components[c.TARGET]:
            t = observation[comp[c.NAME]]
            for prop in self.components[c.PROPS]:
                obs_vec = np.append(obs_vec, t[prop])

        if self.include_current_excitations:
            obs_vec = np.append(obs_vec, js.get(c.EXCITATIONS_STR, []))

        return obs_vec.astype(np.float32)

    def distance_to_target(self, observation):
        diff = 0.0
        for cur, tgt in zip(self.components[c.CURRENT], self.components[c.TARGET]):
            for prop in self.components[c.PROPS]:
                p_cur = np.asarray(observation[cur[c.NAME]][prop])
                p_tgt = np.asarray(observation[tgt[c.NAME]][prop])
                diff += np.linalg.norm(p_cur - p_tgt)
        return diff

    def wrap_action(self, action):
        if self.incremental_actions:
            current = np.array(self.get_excitations_dict())
            return np.clip(action + current, c.LOW_EXCITATION, c.HIGH_EXCITATION)
        return action

    def _sleep_sim(self, seconds):
        """Wait for `seconds` of simulation time to pass (with a small real-time yield)."""
        start = self.net.send_msg(request_type=c.GET_STR, message=c.TIME)
        while self.net.send_msg(request_type=c.GET_STR, message=c.TIME) - start < seconds:
            _time.sleep(0.001)  # yield CPU so the simulation thread can run
