# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_cnn_lstm_model.py
# time: 2019-05-31 19:05

import unittest

import os
import time
import tempfile
import numpy as np
import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.labeling import CNN_LSTM_Model
from kashgari.macros import DATA_PATH

from tensorflow.python.keras.utils import get_file

valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

sample_w2v_path = get_file('sample_w2v.txt',
                           "http://s3.bmio.net/kashgari/sample_w2v.txt",
                           cache_dir=DATA_PATH)

w2v_embedding = WordEmbedding(sample_w2v_path, task=kashgari.LABELING)
w2v_embedding_variable_len = WordEmbedding(sample_w2v_path, task=kashgari.LABELING, sequence_length='variable')


class TestCNN_LSTM_Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = CNN_LSTM_Model

    def test_basic_use_build(self):
        model = self.model_class()
        model.fit(valid_x, valid_y, valid_x, valid_y, epochs=1)
        model.predict_entities(valid_x[:5])
        model.evaluate(valid_x[:100], valid_y[:100])

        res = model.predict(valid_x[:20])
        assert len(res) == 20

        for i in range(5):
            assert len(res[i]) == min(model.embedding.sequence_length, len(valid_x[i]))
        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)

        pd_model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        kashgari.utils.convert_to_saved_model(model,
                                              pd_model_path)

        new_model = kashgari.utils.load_model(model_path)
        new_res = new_model.predict(valid_x[:20])
        assert np.array_equal(new_res, res)

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
