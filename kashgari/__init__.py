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
os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras_bert
from kashgari.macros import TaskType, config

custom_objects = keras_bert.get_custom_objects()
CLASSIFICATION = TaskType.CLASSIFICATION
LABELING = TaskType.LABELING
SCORING = TaskType.SCORING

from kashgari.__version__ import __version__

from kashgari import layers
from kashgari import corpus
from kashgari import embeddings
from kashgari import macros
from kashgari import processors
from kashgari import tasks
from kashgari import utils
from kashgari import callbacks

from kashgari import migration

migration.show_migration_guide()
