# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_blstm_attention_model.py
# time: 2019-05-31 19:15

from tests.labeling.test_cnn_lstm_model import TestCNNLSTMModel
from kashgari.tasks.labeling.experimental import BLSTMAttentionModel


class TestBLSTMAttentionModel(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model_class = BLSTMAttentionModel

    def test_variable_length_model(self):
        pass


if __name__ == "__main__":
    print("Hello world")
