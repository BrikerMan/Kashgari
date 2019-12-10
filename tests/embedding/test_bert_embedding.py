# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bert_embedding.py
# time: 2019-05-31 19:32
import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.macros import DATA_PATH

from tensorflow.python.keras.utils import get_file

import tests.embedding.test_bare_embedding as base


class TestBERTEmbedding(base.TestBareEmbedding):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = BERTEmbedding
        bert_path = get_file('bert_sample_model',
                             "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                             cache_dir=DATA_PATH,
                             untar=True)
        cls.config = {
            'model_folder': bert_path
        }

    def test_variable_length_embed(self):
        with self.assertRaises(Exception):
            self.embedding_class(task=kashgari.CLASSIFICATION,
                                 sequence_length='variable',
                                 **self.config)


if __name__ == "__main__":
    print("Hello world")
