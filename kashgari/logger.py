# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: logger.py
# time: 11:43 下午

import logging

logger = logging.Logger('kashgari', level='DEBUG')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
logger.addHandler(stream_handler)

if __name__ == "__main__":
    pass
