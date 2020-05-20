# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_gru_model.py
# time: 12:35 上午

import unittest

import tests.test_labeling.test_bi_lstm_model as base
from kashgari.tasks.labeling import BiGRU_Model


class TestBiGRU_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiGRU_Model
        
    def test_basic_use(self):
        super(TestBiGRU_Model, self).test_basic_use()


if __name__ == "__main__":
    pass
