# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_kmax_cnn.py
# time: 2019-06-26 17:32
import tests.classification.test_bi_lstm as base
from kashgari.tasks.classification import KMax_CNN_Model


class TestKMax_CNN_Model(base.TestBi_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = KMax_CNN_Model


if __name__ == "__main__":
    print("Hello world")
