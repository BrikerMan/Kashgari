# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_cnn.py
# time: 21:35

import tests.classification.test_bi_lstm as base
from kashgari.tasks.classification import CNN_Model


class TestBiGRUModel(base.TestBi_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = CNN_Model


if __name__ == "__main__":
    print("hello, world")