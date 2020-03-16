# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bare_embedding.py
# time: 2:29 下午

import unittest
from kashgari.corpus import SMP2018ECDTCorpus, ChineseDailyNerCorpus
from kashgari.embeddings import BareEmbedding


class TestBareEmbedding(unittest.TestCase):

    def test_base_cases(self):
        x, y = SMP2018ECDTCorpus.load_data()
        embedding = BareEmbedding()
        embedding.build(x, y)
        res = embedding.embed(x[:10])
        max_len = max([len(i) for i in x[:10]])
        assert res.shape == (10, max_len, 100)

        embedding.set_sequence_length(30)
        res = embedding.embed(x[:2])
        assert res.shape == (2, 30, 100)

        x, y = ChineseDailyNerCorpus.load_data()
        embedding2 = BareEmbedding(sequence_length=25, embedding_size=32)
        embedding2.build(x, y)
        res = embedding2.embed(x[:2])
        assert res.shape == (2, 25, 32)


if __name__ == "__main__":
    unittest.main()
