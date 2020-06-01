# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_word_embedding.py
# time: 2:55 下午

import unittest

from tensorflow.keras.utils import get_file

from kashgari.embeddings import WordEmbedding
from kashgari.macros import DATA_PATH
from tests.test_embeddings.test_bare_embedding import TestBareEmbedding


class TestWordEmbedding(TestBareEmbedding):

    def build_embedding(self):
        sample_w2v_path = get_file('sample_w2v.txt',
                                   "http://s3.bmio.net/kashgari/sample_w2v.txt",
                                   cache_dir=DATA_PATH)
        embedding = WordEmbedding(sample_w2v_path)
        return embedding


if __name__ == '__main__':
    unittest.main()
