# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_blstm_attention_model.py
# time: 2019-05-31 19:15

from tests.labeling import test_cnn_lstm_model as base
from kashgari.tasks.labeling.experimental import BLSTMAttentionModel


class TestBLSTMAttentionModel(base.TestCNN_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = BLSTMAttentionModel

    def test_variable_length_model(self):
        pass


if __name__ == "__main__":
    print("Hello world")
