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
import kashgari
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.tasks.labeling import BiLSTM_Model
from tests.test_macros import TestMacros


class TestBiLSTM_Model(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiLSTM_Model

    def test_basic_use(self):
        model = self.TASK_MODEL_CLASS()
        train_x, train_y = TestMacros.load_labeling_corpus()

        model.fit(train_x,
                  train_y,
                  epochs=self.EPOCH_COUNT)

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        original_y = model.predict(train_x[:20])
        model.save(model_path)
        del model

        new_model = self.TASK_MODEL_CLASS.load_model(model_path)
        new_model.tf_model.summary()
        new_y = new_model.predict(train_x[:20])
        assert new_y == original_y

        report = new_model.evaluate(train_x, train_y)
        print(report)

    def test_with_word_embedding(self):
        w2v_embedding = WordEmbedding(TestMacros.w2v_path)
        model = self.TASK_MODEL_CLASS(embedding=w2v_embedding, sequence_length=120)
        train_x, train_y = TestMacros.load_labeling_corpus()
        valid_x, valid_y = train_x, train_y

        model.fit(train_x,
                  train_y,
                  x_validate=valid_x,
                  y_validate=valid_y,
                  epochs=self.EPOCH_COUNT)


if __name__ == '__main__':
    unittest.main()

