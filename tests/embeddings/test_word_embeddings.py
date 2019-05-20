# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_word_embeddings.py
# time: 2019-05-20 18:22


import unittest

from tensorflow.python.keras import utils

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import WordEmbedding

SAMPLE_WORD2VEC_URL = 'http://storage.eliyar.biz/embedding/word2vec/sample_w2v.txt'


class TestWordEmbedding(unittest.TestCase):

    def test_embed(self):
        sample_w2v_path = utils.get_file('sample_w2v.txt', SAMPLE_WORD2VEC_URL)
        embedding = WordEmbedding(sample_w2v_path)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.prepare_for_labeling(valid_x, valid_y)

        assert embedding.embed(['我', '想', '看']).shape == (97, 100)

        assert embedding.batch_embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 97, 100)

    def test_variable_length_embed(self):
        sample_w2v_path = utils.get_file('sample_w2v.txt', SAMPLE_WORD2VEC_URL)
        embedding = WordEmbedding(sample_w2v_path, sequence_length='variable')

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.prepare_for_labeling(valid_x, valid_y)

        assert embedding.embed(['我', '想', '看']).shape == (3, 100)

        assert embedding.embed(['Hello', 'World']).shape == (2, 100)

        assert embedding.batch_embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 4, 100)


if __name__ == "__main__":
    print("Hello world")
