# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bare_embeddings.py
# time: 2019-05-20 18:54

import unittest
import numpy as np

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import BareEmbedding, WordEmbedding, BERTEmbedding
from kashgari.embeddings import NumericFeaturesEmbedding, StackedEmbedding
from kashgari.processors import ClassificationProcessor, LabelingProcessor
from kashgari.tasks.labeling import BLSTMModel
from kashgari.macros import DATA_PATH

from tensorflow.python.keras.utils import get_file


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
        assert embedding.embed_one(['我', '想', '看']).shape == (20, 55)


class TestWordEmbedding(TestBareEmbedding):
    @classmethod
    def setUpClass(cls):
        sample_w2v_path = get_file('sample_w2v.txt',
                                   "https://storage.googleapis.com/kashgari/sample_w2v.txt",
                                   cache_dir=DATA_PATH)

        cls.embedding_class = WordEmbedding

        cls.config = {
            'w2v_path': sample_w2v_path
        }
        cls.embedding_size = 100

    def test_init_with_processor(self):
        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

        processor = ClassificationProcessor()
        processor.analyze_corpus(valid_x, valid_y)

        embedding = self.embedding_class(sequence_length=20,
                                         processor=processor,
                                         **self.config)
        embedding.analyze_corpus(valid_x, valid_y)
        assert embedding.embed_one(['我', '想', '看']).shape == (20, self.embedding_size)


class TestBERTEmbedding(TestBareEmbedding):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = BERTEmbedding
        bert_path = get_file('bert_sample_model',
                             "https://storage.googleapis.com/kashgari/bert_sample_model.tar.bz2",
                             cache_dir=DATA_PATH)
        cls.config = {
            'bert_path': bert_path
        }

    def test_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         **self.config)

        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (15, 16)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 15, 16)

        embedding = self.embedding_class(task=kashgari.LABELING,
                                         sequence_length=10,
                                         **self.config)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (10, 16)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 10, 16)

    def test_variable_length_embed(self):
        with self.assertRaises(Exception):
            self.embedding_class(task=kashgari.CLASSIFICATION,
                                 sequence_length='variable',
                                 **self.config)

    def test_init_with_processor(self):
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        processor = LabelingProcessor()
        processor.analyze_corpus(valid_x, valid_y)

        embedding = self.embedding_class(sequence_length=11,
                                         processor=processor,
                                         **self.config)
        embedding.analyze_corpus(valid_x, valid_y)
        if self.embedding_class == BERTEmbedding:
            assert embedding.embed_one(['我', '想', '看']).shape == (11, 16)
        else:
            assert embedding.embed_one(['我', '想', '看']).shape == (11, 4)


class TestNumericFeaturesEmbedding(unittest.TestCase):

    def test_embedding(self):
        embed = NumericFeaturesEmbedding(2,
                                         feature_name='is_bold',
                                         sequence_length=10,
                                         embedding_size=30)
        embed.embed_model.summary()
        assert embed.embed_one([1, 2]).shape == (10, 30)
        assert embed.embed([[1, 2]]).shape == (1, 10, 30)


class TestStackedEmbedding(unittest.TestCase):

    def test_embedding(self):
        from kashgari.corpus import ChineseDailyNerCorpus
        from kashgari.embeddings import BareEmbedding, NumericFeaturesEmbedding

        text, label = ChineseDailyNerCorpus.load_data()
        is_bold = np.random.randint(1, 3, (len(text), 12))

        text_embedding = BareEmbedding(task=kashgari.LABELING,
                                       sequence_length=12)
        num_feature_embedding = NumericFeaturesEmbedding(2,
                                                         'is_bold',
                                                         sequence_length=12)

        stack_embedding = StackedEmbedding([text_embedding, num_feature_embedding])
        stack_embedding.analyze_corpus((text, is_bold), label)

        r = stack_embedding.embed((text[:3], is_bold[:3]))
        assert r.shape == (3, 12, 116)

    def test_training(self):
        import kashgari
        from kashgari.embeddings import NumericFeaturesEmbedding, BareEmbedding, StackedEmbedding

        text = ['NLP', 'Projects', 'Project', 'Name', ':']
        start_of_p = [1, 2, 1, 2, 2]
        bold = [1, 1, 1, 1, 2]
        center = [1, 1, 2, 2, 2]
        label = ['B-Category', 'I-Category', 'B-ProjectName', 'I-ProjectName', 'I-ProjectName']

        text_list = [text] * 300
        start_of_p_list = [start_of_p] * 300
        bold_list = [bold] * 300
        center_list = [center] * 300
        label_list = [label] * 300

        # You can use WordEmbedding or BERTEmbedding for your text embedding
        SEQUENCE_LEN = 100
        text_embedding = BareEmbedding(task=kashgari.LABELING, sequence_length=SEQUENCE_LEN)
        start_of_p_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                        feature_name='start_of_p',
                                                        sequence_length=SEQUENCE_LEN)

        bold_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                  feature_name='bold',
                                                  sequence_length=SEQUENCE_LEN,
                                                  embedding_size=10)

        center_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                    feature_name='center',
                                                    sequence_length=SEQUENCE_LEN)

        # first one must be the text, embedding
        stack_embedding = StackedEmbedding([
            text_embedding,
            start_of_p_embedding,
            bold_embedding,
            center_embedding
        ])

        x = (text_list, start_of_p_list, bold_list, center_list)
        y = label_list
        stack_embedding.analyze_corpus(x, y)

        model = BLSTMModel(embedding=stack_embedding)
        model.build_model(x, y)

        model.fit(x, y, epochs=2)


if __name__ == "__main__":
    print("Hello world")
