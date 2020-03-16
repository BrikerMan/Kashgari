# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_lstm_model.py
# time: 4:41 下午

import os
import tempfile
import time
import unittest

from kashgari.embeddings import WordEmbedding
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.tasks.labeling import BiLSTM_Model
from kashgari.utils import load_model
from tests.test_macros import TestMacros


class TestBiLSTM_Model(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiLSTM_Model

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.w2v_embedding

    def test_basic_use(self):
        model = self.TASK_MODEL_CLASS()
        train_x, train_y = TestMacros.load_labeling_corpus('custom_1')

        model.fit(train_x,
                  train_y,
                  epochs=self.EPOCH_COUNT)

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        original_y = model.predict(train_x[:20])
        model.save(model_path)
        del model
        new_model = load_model(model_path)
        new_model.tf_model.summary()
        new_y = new_model.predict(train_x[:20])
        assert new_y == original_y

    def test_with_word_embedding(self):
        w2v_embedding = WordEmbedding(TestMacros.w2v_path, sequence_length=80)
        w2v_embedding.set_sequence_length(120)
        model = self.TASK_MODEL_CLASS(embedding=w2v_embedding)
        train_x, train_y = TestMacros.load_labeling_corpus('custom_1')
        valid_x, valid_y = train_x, train_y

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)


if __name__ == '__main__':
    unittest.main()

