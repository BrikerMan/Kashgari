# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bare_embeddings.py
# time: 2019-05-20 18:54

import os
import unittest

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import BareEmbedding, WordEmbedding, BertEmbedding
from kashgari.pre_processors import ClassificationProcessor


class TestBareEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = BareEmbedding
        cls.config = {}
        cls.embedding_size = 150

    def test_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         embedding_size=self.embedding_size,
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
                                         embedding_size=self.embedding_size,
                                         **self.config)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (10, self.embedding_size)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 10, embedding.embedding_size)

    def test_variable_length_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         sequence_length='variable',
                                         embedding_size=self.embedding_size,
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
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        processor = ClassificationProcessor()
        processor.analyze_corpus(valid_x, valid_y)

        embedding = self.embedding_class(sequence_length=20,
                                         processor=processor,
                                         embedding_size=self.embedding_size,
                                         **self.config)
        assert embedding.embed_one(['我', '想', '看']).shape == (20, self.embedding_size)


class TestWordEmbedding(TestBareEmbedding):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = WordEmbedding
        cls.config = {
            'w2v_path': os.path.join(kashgari.utils.get_project_path(), 'tests/test-data/sample_w2v.txt')
        }
        cls.embedding_size = 100

    def test_multi_input_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         sequence_length=(24, 24),
                                         **self.config)

        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        data1 = [
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]

        data2 = [
            ['你', '是', '谁'],
            ['好', '好', '玩', '啊'],
            ['你', '好']
        ]

        embed_res = embedding.embed((data1, data2))

        assert embed_res[0].shape == embed_res[1].shape
        assert embed_res[0].shape == (24, 200)

    def test_init_with_processor(self):
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        processor = ClassificationProcessor()
        processor.analyze_corpus(valid_x, valid_y)

        embedding = self.embedding_class(sequence_length=20,
                                         processor=processor,
                                         embedding_size=self.embedding_size,
                                         **self.config)
        embedding.analyze_corpus(valid_x, valid_y)
        assert embedding.embed_one(['我', '想', '看']).shape == (20, self.embedding_size)


class TestBertEmbedding(TestBareEmbedding):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = BertEmbedding
        cls.config = {
            'bert_path': os.path.join(kashgari.utils.get_project_path(), 'tests/test-data/bert')
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

        assert embedding.embed_one(['我', '想', '看']).shape == (10, 4)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 10, 4)

    def test_multi_input_embed(self):
        embedding = BertEmbedding(task=kashgari.CLASSIFICATION,
                                  sequence_length=(12, 12),
                                  **self.config)

        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        data1 = [
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]

        data2 = [
            ['你', '是', '谁'],
            ['好', '好', '玩', '啊'],
            ['你', '好']
        ]

        embed_res = embedding.embed((data1, data2))
        assert embed_res[0].shape == embed_res[1].shape
        assert embed_res[0].shape == (12, 4)

    def test_variable_length_embed(self):
        with self.assertRaises(Exception) as context:
            self.embedding_class(task=kashgari.CLASSIFICATION,
                                 sequence_length='variable',
                                 **self.config)

    def test_init_with_processor(self):
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        processor = ClassificationProcessor()
        processor.analyze_corpus(valid_x, valid_y)

        embedding = self.embedding_class(sequence_length=11,
                                         processor=processor,
                                         **self.config)
        embedding.analyze_corpus(valid_x, valid_y)
        assert embedding.embed_one(['我', '想', '看']).shape == (11, 4)


if __name__ == "__main__":
    print("Hello world")
