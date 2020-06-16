# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: logger.py
# time: 11:43 下午

import os
import logging

logger = logging.Logger('kashgari', level='DEBUG')
stream_handler = logging.StreamHandler()

if os.environ.get('KASHGARI_DEV') == 'True':
    log_format = '%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s'
else:
    log_format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'

stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

if __name__ == "__main__":
    pass
