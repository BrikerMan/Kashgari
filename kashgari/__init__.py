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
from typing import Dict, Any

os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

custom_objects: Dict[str, Any] = {}

from kashgari.__version__ import __version__
from kashgari.macros import config
from kashgari import layers
from kashgari import corpus
from kashgari import embeddings
from kashgari import macros
from kashgari import processors
from kashgari import tasks
from kashgari import utils

from kashgari.utils.dependency_check import dependency_check

custom_objects = layers.resigter_custom_layers(custom_objects)

dependency_check()
