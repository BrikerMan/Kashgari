# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_avcnn.py
# time: 2019-06-26 17:32

import tests.classification.test_bi_lstm as base
from kashgari.tasks.classification import AVCNN_Model


class TestAVCNN_Model(base.TestBi_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = AVCNN_Model

    def test_basic_use(self):
        super(TestAVCNN_Model, self).test_basic_use()


if __name__ == "__main__":
    print("Hello world")
