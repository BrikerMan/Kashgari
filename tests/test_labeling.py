# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_labeling.py
# time: 2019-05-20 19:03

import unittest

from tensorflow.python.keras import utils

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.labeling import CNNLSTMModel, BLSTMModel

SAMPLE_WORD2VEC_URL = 'http://storage.eliyar.biz/embedding/word2vec/sample_w2v.txt'

valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

sample_w2v_path = utils.get_file('sample_w2v.txt', SAMPLE_WORD2VEC_URL)

w2v_embedding = WordEmbedding(sample_w2v_path, task=kashgari.LABELING)
w2v_embedding_variable_len = WordEmbedding(sample_w2v_path, task=kashgari.LABELING, sequence_length='variable')


class TestCNNLSTMModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = CNNLSTMModel

    def test_basic_use_build(self):
        model = self.model_class()
        model.fit(valid_x, valid_y, valid_x, valid_y, epochs=1)
        assert True

    def test_w2v_model(self):
        model = self.model_class(embedding=w2v_embedding)
        model.fit(valid_x, valid_y, epochs=1)
        assert True

    def test_variable_length_model(self):
        hyper_params = self.model_class.get_default_hyper_parameters()

        for layer, config in hyper_params.items():
            for key, value in config.items():
                if isinstance(value, int):
                    hyper_params[layer][key] = value + 15

        model = self.model_class(embedding=w2v_embedding_variable_len,
                                 hyper_parameters=hyper_params)
        model.fit(valid_x, valid_y, epochs=1)
        assert True

    def test_multi_input(self):
        old_fashion_model = self.model_class()
        old_fashion_model.fit((valid_x, valid_x), valid_y, epochs=1)


class TestBLSTMModel(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model_class = BLSTMModel


if __name__ == "__main__":
    print("Hello world")
