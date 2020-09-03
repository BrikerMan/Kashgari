#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/2 2:10 下午
# File    : test_bi_lstm_crf_model.py
# Project : Kashgari

import pytest
import kashgari
import tests.test_labeling.test_bi_lstm_model as base
from kashgari.tasks.labeling import BiLSTM_CRF_Model


@pytest.mark.skipif(kashgari.__version__ == "2.0.0.alpha02",
                    reason="Skip in 2.0.0.alpha02, will fix on final release")
class TestBiLSTMGRU_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiLSTM_CRF_Model
