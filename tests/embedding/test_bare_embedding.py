# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bare_embedding.py
# time: 2019-05-31 19:29

import unittest

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import BareEmbedding
from kashgari.embeddings import BERTEmbedding
from kashgari.processors import ClassificationProcessor


class TestBareEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = BareEmbedding
        cls.config = {
            'embedding_size': 150
        }

    def test_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         **self.config)

        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (15, embedding.embedding_size)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 15, embedding.embedding_size)

        embedding = self.embedding_class(task=kashgari.LABELING,
                                         sequence_length=10,
                                         **self.config)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (10, embedding.embedding_size)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 10, embedding.embedding_size)

    def test_variable_length_embed(self):
        if self.embedding_class is BareEmbedding:
            self.config['embedding_size'] = 128

        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         sequence_length='variable',
                                         **self.config)

        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (3, embedding.embedding_size)

        assert embedding.embed_one(['Hello', 'World']).shape == (2, embedding.embedding_size)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 4, embedding.embedding_size)

    def test_init_with_processor(self):
        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

        processor = ClassificationProcessor()
        processor.analyze_corpus(valid_x, valid_y)
        if self.embedding_class is BareEmbedding:
            self.config['embedding_size'] = 55

        embedding = self.embedding_class(sequence_length=20,
                                         processor=processor,
                                         **self.config)
        if self.embedding_class is BERTEmbedding:
            seq_len = 16
        else:
            seq_len = 20

        assert embedding.embed_one(['我', '想', '看']).shape == (seq_len, embedding.embedding_size)


if __name__ == "__main__":
    print("Hello world")
