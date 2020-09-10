#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/2 2:09 下午
# File    : test_bi_gru_crf_model.py
# Project : Kashgari
from distutils.version import LooseVersion

import pytest
import tensorflow as tf

import tests.test_labeling.test_bi_lstm_model as base
from kashgari.tasks.labeling import BiGRU_CRF_Model


@pytest.mark.skipif(LooseVersion(tf.__version__) < '2.2.0',
                    reason="The KConditionalRandomField requires TensorFlow 2.2.x version or higher.")
class TestBiGRU_CRF_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiGRU_CRF_Model
