# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bare_embeddings.py
# time: 2019-05-20 18:54

import unittest

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import BareEmbedding
from kashgari.pre_processors import PreProcessor


class TestWordEmbedding(unittest.TestCase):

    def test_embed(self):
        embedding = BareEmbedding(embedding_size=150)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.prepare_for_labeling(valid_x, valid_y)

        assert embedding.embed(['我', '想', '看']).shape == (97, 150)

        assert embedding.batch_embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 97, 150)

        embedding = BareEmbedding(sequence_length=50, embedding_size=150)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.prepare_for_labeling(valid_x, valid_y)

        assert embedding.embed(['我', '想', '看']).shape == (50, 150)

        assert embedding.batch_embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 50, 150)

    def test_variable_length_embed(self):
        embedding = BareEmbedding(sequence_length='variable', embedding_size=200)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.prepare_for_labeling(valid_x, valid_y)

        assert embedding.embed(['我', '想', '看']).shape == (3, 200)

        assert embedding.embed(['Hello', 'World']).shape == (2, 200)

        assert embedding.batch_embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 4, 200)

    def test_init_with_processor(self):
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        processor = PreProcessor()
        processor.prepare_labeling_dicts_if_need(valid_x, valid_y)

        embedding = BareEmbedding(sequence_length=20, processor=processor)
        assert embedding.embed(['我', '想', '看']).shape == (20, 100)


if __name__ == "__main__":
    print("Hello world")
