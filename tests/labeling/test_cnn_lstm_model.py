# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_cnn_lstm_model.py
# time: 2019-05-31 19:05

import os
import tempfile
import time
import unittest

import numpy as np
from tensorflow.python.keras.utils import get_file

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import WordEmbedding
from kashgari.macros import DATA_PATH
from kashgari.tasks.labeling import CNN_LSTM_Model
from tests.corpus import NERCorpus

valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

sample_w2v_path = get_file('sample_w2v.txt',
                           "http://s3.bmio.net/kashgari/sample_w2v.txt",
                           cache_dir=DATA_PATH)


class TestCNN_LSTM_Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = CNN_LSTM_Model

    def test_basic_use_build(self):
        x, y = NERCorpus.load_corpus()

        model = self.model_class()
        model.fit(x, y, x, y, epochs=1)
        model.predict_entities(x[:5])
        model.evaluate(x, y)

        res = model.predict(x[:20])
        assert len(res) == min(len(x), 20)

        for i in range(5):
            assert len(res[i]) == min(model.embedding.sequence_length, len(x[i]))

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)

        pd_model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        kashgari.utils.convert_to_saved_model(model,
                                              pd_model_path)

        new_model = kashgari.utils.load_model(model_path)
        new_res = new_model.predict(x[:20])
        assert np.array_equal(new_res, res)

        new_model.compile_model()
        model.fit(x, y, x, y, epochs=1)

    def test_fit_without_generator(self):
        x, y = NERCorpus.load_corpus('custom_2')
        model = self.model_class()
        model.fit_without_generator(x, y, x, y, epochs=2)

    def test_w2v_model(self):
        x, y = NERCorpus.load_corpus()
        w2v_embedding = WordEmbedding(sample_w2v_path, task=kashgari.LABELING)
        model = self.model_class(embedding=w2v_embedding)
        try:
            model.fit(x, y, x, y, epochs=1)
            model.evaluate(x, y)
            assert True
        except Exception as e:
            print(model.label2idx)
            raise e

    def test_variable_length_model(self):
        x, y = NERCorpus.load_corpus('custom_2')
        hyper_params = self.model_class.get_default_hyper_parameters()

        for layer, config in hyper_params.items():
            for key, value in config.items():
                if isinstance(value, int):
                    hyper_params[layer][key] = value + 15

        w2v_embedding_variable_len = WordEmbedding(sample_w2v_path,
                                                   task=kashgari.LABELING,
                                                   sequence_length='variable')
        model = self.model_class(embedding=w2v_embedding_variable_len,
                                 hyper_parameters=hyper_params)
        try:
            model.fit(x, y, epochs=1)
            model.evaluate(x, y)
            assert True
        except Exception as e:
            print(model.label2idx)
            raise e
