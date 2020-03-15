# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 1:57 下午

import unittest

from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.classification import BiLSTM_Model
from tests.test_macros import TestMacros


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
        train_x, train_y = SMP2018ECDTCorpus.load_data()
        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)

    def test_with_word_embedding(self):
        model = self.TASK_MODEL_CLASS(embedding=self.w2v_embedding)
        train_x, train_y = SMP2018ECDTCorpus.load_data()
        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)


if __name__ == '__main__':
    unittest.main()
