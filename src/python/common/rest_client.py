import json
import logging

import requests

from common import constants as c

logger = logging.getLogger()

# If ArtiSynth does not reply within this many seconds, treat it as hung.
REQUEST_TIMEOUT = 30


class RestClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    @staticmethod
    def get_url(ip, port, message):
        return f'http://{ip}:{port}/{message}'

    @staticmethod
    def server_is_alive(ip, port):
        url = RestClient.get_url(ip, port, '')
        try:
            response = requests.get(url, timeout=2)
            return response.ok
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logger.info('ArtiSynth server not reachable at %s', url)
            return False

    def send_msg(self, obj=None, request_type=c.GET_STR, message=''):
        url = RestClient.get_url(self.ip, self.port, message)
        try:
            if request_type == c.GET_STR:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
            elif request_type == c.POST_STR:
                response = requests.post(url, json=obj, timeout=REQUEST_TIMEOUT)
            else:
                raise ValueError(f'Unknown request_type: {request_type}')

            if response.ok:
                return json.loads(response.content.decode())
            logger.warning('Bad response %s from %s', response.status_code, url)
            return {}

        except requests.exceptions.Timeout:
            logger.error('ArtiSynth timed out after %ds on %s', REQUEST_TIMEOUT, url)
            raise
        except requests.exceptions.ConnectionError as err:
            logger.error('Connection error to %s: %s', url, err)
            raise
