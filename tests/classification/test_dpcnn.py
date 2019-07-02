# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_dpcnn.py
# time: 2019-07-02 20:45

import tests.classification.test_bi_lstm as base
from kashgari.tasks.classification import DPCNN_Model


class TestDPCNN_Model(base.TestBi_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = DPCNN_Model


if __name__ == "__main__":
    print("Hello world")
