# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_lstm_model.py
# time: 4:41 下午

import logging
import unittest

from kashgari.embeddings import WordEmbedding
from kashgari.tasks.labeling import BiLSTM_Model
from tests.test_macros import TestMacros


logging.basicConfig(level='DEBUG')


class TestBiLSTM_Model(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiLSTM_Model
        cls.w2v_embedding = WordEmbedding(TestMacros.w2v_path)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.w2v_embedding

    def test_basic_use(self):
        model = self.TASK_MODEL_CLASS()
        train_x, train_y = TestMacros.load_labeling_corpus()

        model.fit(train_x,
                  train_y,
                  epochs=self.EPOCH_COUNT)

    def test_with_word_embedding(self):
        self.w2v_embedding.set_sequence_length(120)
        model = self.TASK_MODEL_CLASS(embedding=self.w2v_embedding)
        train_x, train_y = TestMacros.load_labeling_corpus()
        valid_x, valid_y = train_x, train_y

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)


if __name__ == '__main__':
    unittest.main()

