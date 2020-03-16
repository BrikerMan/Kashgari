# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_word_embedding.py
# time: 2:55 下午

import unittest
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import WordEmbedding
from kashgari.macros import DATA_PATH

from tensorflow.keras.utils import get_file


class TestWordEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_w2v_path = get_file('sample_w2v.txt',
                                   "http://s3.bmio.net/kashgari/sample_w2v.txt",
                                   cache_dir=DATA_PATH)

        cls.embedding_class = WordEmbedding

        cls.w2v_path = sample_w2v_path
        cls.embedding_size = 100

    def test_base_cases(self):
        x, y = SMP2018ECDTCorpus.load_data()
        embedding = WordEmbedding(self.w2v_path)
        embedding.build(x, y)
        res = embedding.embed(x[:10])
        max_len = max([len(i) for i in x[:10]])
        assert res.shape == (10, max_len, 100)

        embedding.set_sequence_length(30)
        res = embedding.embed(x[:2])
        assert res.shape == (2, 30, 100)


if __name__ == '__main__':
    unittest.main()
