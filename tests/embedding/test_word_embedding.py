# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_word_embedding.py
# time: 2019-05-31 19:31
import os
import unittest
import numpy as np

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import BareEmbedding, WordEmbedding
from kashgari.embeddings import BERTEmbedding, GPT2Embedding
from kashgari.embeddings import NumericFeaturesEmbedding, StackedEmbedding
from kashgari.processors import ClassificationProcessor, LabelingProcessor
from kashgari.tasks.labeling import BLSTMModel
from kashgari.macros import DATA_PATH

from tensorflow.python.keras.utils import get_file

from tests.embedding.test_bare_embedding import TestBareEmbedding


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


if __name__ == "__main__":
    print("Hello world")
