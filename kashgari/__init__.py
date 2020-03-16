# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py
@time: 2019-05-17 11:15

"""

import os
from kashgari.__version__ import __version__
import logging

os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

custom_objects = {}
logger = logging.Logger('kashgari', level='DEBUG')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
logger.addHandler(stream_handler)

from kashgari.macros import config

from kashgari import layers
from kashgari import corpus
from kashgari import embeddings
from kashgari import macros
from kashgari import processors
from kashgari import tasks
from kashgari import utils
