# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: classification.py
# time: 2019-05-22 12:39
import os
import logging
import unittest
import numpy as np

import time
import kashgari
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import WordEmbedding, BERTEmbedding
from kashgari.tasks.classification import BLSTMModel
from kashgari.macros import DATA_PATH
from tensorflow.python.keras.utils import get_file

valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

bert_path = get_file('bert_sample_model',
                     "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                     cache_dir=DATA_PATH,
                     untar=True)

sample_w2v_path = get_file('sample_w2v.txt',
                           "http://s3.bmio.net/kashgari/sample_w2v.txt",
                           cache_dir=DATA_PATH)

w2v_embedding = WordEmbedding(sample_w2v_path, task=kashgari.CLASSIFICATION)
w2v_embedding_variable_len = WordEmbedding(sample_w2v_path, task=kashgari.CLASSIFICATION, sequence_length='variable')

logging.basicConfig(level=logging.DEBUG)


class TestBertCNN_LSTM_Model(unittest.TestCase):
    def test_bert_model(self):
        embedding = BERTEmbedding(bert_path, task=kashgari.CLASSIFICATION, sequence_length=100)
        model = BLSTMModel(embedding=embedding)
        model.fit(valid_x, valid_y, epochs=1)
        res = model.predict(valid_x[:20])
        assert True

        model_path = os.path.join('./saved_models/',
                                  model.__class__.__module__,
                                  model.__class__.__name__,
                                  str(time.time()))
        model.save(model_path)

        new_model = kashgari.utils.load_model(model_path)
        new_res = new_model.predict(valid_x[:20])
        assert np.array_equal(new_res, res)


class TestCNN_LSTM_Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = BLSTMModel

    def test_basic_use(self):
        model = self.model_class()
        model.fit(valid_x, valid_y, valid_x, valid_y, epochs=1)
        res = model.predict(valid_x[:20])
        assert len(res) == 20
        model_path = os.path.join('./saved_models/',
                                  model.__class__.__module__,
                                  model.__class__.__name__)
        model.save(model_path)

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
                if isinstance(value, bool):
                    pass
                elif isinstance(value, int):
                    hyper_params[layer][key] = value + 15
        model = self.model_class(embedding=w2v_embedding_variable_len,
                                 hyper_parameters=hyper_params)
        model.fit(valid_x, valid_y, epochs=1)
        assert True


if __name__ == "__main__":
    print("Hello world")
