# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py
@time: 2019-01-24

"""
from .classification import *
from .seq_labeling.blstm_model_test import BLSTMModelTest
from .seq_labeling.blstm_crf_model_test import BLSTMCRFModelTest


if __name__ == '__main__':
    print("hello, world")