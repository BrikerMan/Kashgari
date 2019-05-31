# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_numeric_features_embedding.py
# time: 2019-05-31 19:35

import unittest
from kashgari.embeddings import NumericFeaturesEmbedding


class TestNumericFeaturesEmbedding(unittest.TestCase):

    def test_embedding(self):
        embed = NumericFeaturesEmbedding(2,
                                         feature_name='is_bold',
                                         sequence_length=10,
                                         embedding_size=30)
        embed.embed_model.summary()
        assert embed.embed_one([1, 2]).shape == (10, 30)
        assert embed.embed([[1, 2]]).shape == (1, 10, 30)


if __name__ == "__main__":
    print("Hello world")
