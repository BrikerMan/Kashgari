# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_gru_model.py
# time: 3:28 下午

import unittest

import tests.test_classification.test_bi_lstm_model as base
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.classification import BiGRU_Model
from tests.test_macros import TestMacros


class TestBiGRU_Model(base.TestBiLSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 2
        cls.TASK_MODEL_CLASS = BiGRU_Model
        cls.w2v_embedding = WordEmbedding(TestMacros.w2v_path)


if __name__ == "__main__":
    unittest.main()
