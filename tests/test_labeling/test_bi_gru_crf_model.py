# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_gru_crf_model.py
# time: 5:41 下午

import unittest

import tests.test_labeling.test_bi_lstm_model as base
from kashgari.tasks.labeling import BiGRU_CRF_Model


class TestBiGRU_CRF_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiGRU_CRF_Model


if __name__ == "__main__":
    unittest.main()
