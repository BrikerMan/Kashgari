# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py.py
@time: 2019-01-19 11:49

"""
from .base_model import ClassificationModel
from kashgari.tasks.classification.models import BLSTMModel, CNNLSTMModel, CNNModel
from kashgari.tasks.classification.models import AVCNNModel, KMaxCNNModel, RCNNModel, AVRNNModel
from kashgari.tasks.classification.models import DropoutBGRUModel, DropoutAVRNNModel
