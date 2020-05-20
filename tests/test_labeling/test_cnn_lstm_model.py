# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_cnn_lstm_model.py
# time: 5:41 下午


import unittest

import tests.test_labeling.test_bi_lstm_model as base
from kashgari.tasks.labeling import CNN_LSTM_Model


class TestCNN_LSTM_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = CNN_LSTM_Model


if __name__ == "__main__":
    unittest.main()
