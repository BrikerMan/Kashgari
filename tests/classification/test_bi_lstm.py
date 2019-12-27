# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: classification.py
# time: 2019-05-22 12:39
import os
import logging
import tempfile
import unittest
import numpy as np

import time
import kashgari
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import WordEmbedding, BERTEmbedding
from kashgari.tasks.classification import BLSTMModel
from kashgari.macros import DATA_PATH
from kashgari.processors import ClassificationProcessor
from kashgari.embeddings import BareEmbedding

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


sample_train_x = [
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

sample_train_y = [['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['c']]

sample_eval_x = [
    list('语言学是一门关于人类语言的科学研究。'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构研究与意义研究之间存在一个重要的主题划分。'),
    list('语法中包含了词法，句法以及语音。'),
    list('语音学是语言学的一个相关分支，它涉及到语音与非语音声音的实际属性，以及它们是如何发出与被接收到的。'),
    list('与学习语言不同，语言学是研究所有人类语文发展有关的一门学术科目。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

sample_eval_y = [['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['c'], ['b'], ['a']]


class TestBertCNN_LSTM_Model(unittest.TestCase):
    def test_bert_model(self):
        embedding = BERTEmbedding(bert_path, task=kashgari.CLASSIFICATION, sequence_length=100)
        model = BLSTMModel(embedding=embedding)
        model.fit(valid_x, valid_y, epochs=1)
        res = model.predict(valid_x[:20])
        assert True

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)

        new_model = kashgari.utils.load_model(model_path)
        new_res = new_model.predict(valid_x[:20])
        assert np.array_equal(new_res, res)


class TestBi_LSTM_Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = BLSTMModel

    def test_basic_use(self):
        model = self.model_class()
        model.fit(valid_x, valid_y, valid_x, valid_y, epochs=1)
        res = model.predict(valid_x[:20])
        assert len(res) == 20
        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)

        new_model = kashgari.utils.load_model(model_path)
        new_res = new_model.predict(valid_x[:20])
        assert np.array_equal(new_res, res)

        new_model.evaluate(valid_x, valid_y)
        model.predict_top_k_class(valid_x)

    def test_fit_without_generator(self):
        model = self.model_class()
        model.fit_without_generator(valid_x, valid_y, valid_x, valid_y, epochs=2)

    def test_w2v_model(self):
        model = self.model_class(embedding=w2v_embedding)
        model.fit(valid_x, valid_y, epochs=1)
        assert True

    def test_custom_hyper_params(self):
        hyper_params = self.model_class.get_default_hyper_parameters()

        for layer, config in hyper_params.items():
            for key, value in config.items():
                if isinstance(value, bool):
                    pass
                elif isinstance(value, int):
                    hyper_params[layer][key] = value + 15 if value >= 64 else value
        model = self.model_class(embedding=w2v_embedding_variable_len,
                                 hyper_parameters=hyper_params)
        model.fit(valid_x, valid_y, epochs=1)
        assert True

    def test_multi_label(self):
        p = ClassificationProcessor(multi_label=True)
        embedding = BareEmbedding(task='classification', processor=p)
        model = self.model_class(embedding)
        model.fit(sample_train_x, sample_train_y, epochs=1)
        assert len(p.label2idx) == 3

        model.evaluate(sample_eval_x, sample_eval_y)
        assert isinstance(model.predict(sample_eval_x)[0], tuple)
        report_dict = model.evaluate(sample_eval_x, sample_eval_y, output_dict=True)
        assert isinstance(report_dict, dict)
        res = model.predict(valid_x[:20])
        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)

        new_model = kashgari.utils.load_model(model_path)
        assert res == new_model.predict(valid_x[:20])


if __name__ == "__main__":
    print("Hello world")
