# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_r_cnn.py
# time: 2019-06-27 11:06
import tests.classification.test_bi_lstm as base
from kashgari.tasks.classification import R_CNN_Model


class TestR_CNN_Model(base.TestBi_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = R_CNN_Model

    def test_basic_use(self):
        super(TestR_CNN_Model, self).test_basic_use()


if __name__ == "__main__":
    print("Hello world")
