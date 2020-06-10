# encoding: utf-8

# author: Adline
# contact: gglfxsld@gmail.com
# blog: https://medium.com/@Adline125

# file: test_cnn_attention_model.py
# time: 3:31 下午

import pytest
import unittest
import tensorflow as tf

import tests.test_classification.test_bi_lstm_model as base
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.classification.cnn_attention_model import CNN_Attention_Model
from tests.test_macros import TestMacros


@pytest.mark.xfail(tuple(tf.__version__.split('.')) < tuple('2.1.0'.split('.')),
                   reason='Attention Layer cannot be load and saved in TF 2.0.0')
class TestCnnAttention_Model(base.TestBiLSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = CNN_Attention_Model
        cls.w2v_embedding = WordEmbedding(TestMacros.w2v_path)

    def test_multi_label(self):
        super(TestCnnAttention_Model, self).test_multi_label()


if __name__ == "__main__":
    unittest.main()
