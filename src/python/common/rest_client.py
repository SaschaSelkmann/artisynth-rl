import json
import logging

import requests

from common import constants as c

logger = logging.getLogger()


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
        except requests.exceptions.ConnectionError:
            logger.info('ArtiSynth server not reachable at %s', url)
            return False

    def send_msg(self, obj=None, request_type=c.GET_STR, message=''):
        url = RestClient.get_url(self.ip, self.port, message)
        try:
            if request_type == c.GET_STR:
                response = requests.get(url)
            elif request_type == c.POST_STR:
                response = requests.post(url, json=obj)
            else:
                raise ValueError(f'Unknown request_type: {request_type}')

            if response.ok:
                return json.loads(response.content.decode())
            logger.warning('Bad response %s from %s', response.status_code, url)
            return {}

        except requests.exceptions.ConnectionError as err:
            logger.error('Connection error to %s: %s', url, err)
            raise
