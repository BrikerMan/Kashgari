# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_blstm_model.py
# time: 2019-05-31 19:06

import tests.labeling.test_cnn_lstm_model as base
from kashgari.tasks.labeling import BLSTMModel


class TestBLSTMModel(base.TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model_class = BLSTMModel

    def test_basic_use_build(self):
        super(TestBLSTMModel, self).test_basic_use_build()


if __name__ == "__main__":
    print("Hello world")
