# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_blstm_crf_model.py
# time: 00:38


import tests.labeling.test_cnn_lstm_model as base
from kashgari.tasks.labeling import BLSTMCRFModel


class TestBLSTMModel(base.TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model_class = BLSTMCRFModel

    def test_basic_use_build(self):
        super(TestBLSTMModel, self).test_basic_use_build()


if __name__ == "__main__":
    print("Hello world")