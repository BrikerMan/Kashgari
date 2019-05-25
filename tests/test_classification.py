# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: classification.py
# time: 2019-05-22 12:39

import logging
import unittest

import kashgari
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import WordEmbedding, BERTEmbedding
from kashgari.tasks.classification import BLSTMModel
from kashgari.macros import DATA_PATH
from tensorflow.python.keras.utils import get_file

valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

sample_w2v_path = get_file('sample_w2v.txt',
                           "https://storage.googleapis.com/kashgari/sample_w2v.txt",
                           cache_dir=DATA_PATH)

bert_path = get_file('bert_sample_model',
                            "/Users/brikerman/.kashgari/datasets/bert_sample_model/",
                            cache_dir=DATA_PATH)

w2v_embedding = WordEmbedding(sample_w2v_path, task=kashgari.CLASSIFICATION)
w2v_embedding_variable_len = WordEmbedding(sample_w2v_path, task=kashgari.CLASSIFICATION, sequence_length='variable')

logging.basicConfig(level=logging.DEBUG)


class TestBertCNNLSTMModel(unittest.TestCase):
    def test_bert_model(self):
        embedding = BERTEmbedding(bert_path, kashgari.CLASSIFICATION, sequence_length=100)
        model = BLSTMModel(embedding=embedding)
        model.fit(valid_x, valid_y, epochs=1)
        assert True


class TestCNNLSTMModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = BLSTMModel

    def test_basic_use(self):
        model = self.model_class()
        model.fit(valid_x, valid_y, valid_x, valid_y, epochs=1)
        res = model.predict(valid_x[:5])
        assert len(res) == 5
        model.evaluate(valid_x[:100], valid_y[:100])

        # model_path = os.path.join('./models/', model.info()['task'], self.model_class.__architect_name__)
        # model.save(model_path)
        # new_model = kashgari.utils.load_model(model_path)
        # new_res = new_model.predict(valid_x[:5])
        # assert np.array_equal(new_res, res)

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
