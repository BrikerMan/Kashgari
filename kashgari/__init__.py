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

from kashgari import layers
from kashgari import corpus
from kashgari import embeddings
from kashgari import macros
from kashgari import processors
from kashgari import tasks
from kashgari.version import __version__
from kashgari import utils

from kashgari.macros import TaskType

CLASSIFICATION = TaskType.CLASSIFICATION
LABELING = TaskType.LABELING
