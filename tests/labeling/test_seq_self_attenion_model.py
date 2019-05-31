# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_seq_self_attenion_model.py
# time: 2019-05-31 19:15

from tests.labeling.test_cnn_lstm_model import TestCNNLSTMModel
from kashgari.tasks.labeling.experimental import SeqSelfAttentionModel


class TestSeqSelfAttentionModel(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model_class = SeqSelfAttentionModel

    def test_variable_length_model(self):
        pass


if __name__ == "__main__":
    print("Hello world")
