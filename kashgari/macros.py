# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: macros.py
# time: 12:37 下午

import os
import logging
from pathlib import Path
import tensorflow as tf

DATA_PATH = os.path.join(str(Path.home()), '.kashgari')

Path(DATA_PATH).mkdir(exist_ok=True, parents=True)


class Config(object):

    def __init__(self):
        self._log_level = False

    @property
    def log_level(self):
        return self._log_level

    @property
    def logger(self):
        return logging.getLogger('kashgari')

    @log_level.setter
    def log_level(self, value):
        self._log_level = value
        self.logger.setLevel(level=value)

    def to_dict(self):
        return {
            'verbose': self._log_level
        }


config = Config()

if __name__ == "__main__":
    pass
