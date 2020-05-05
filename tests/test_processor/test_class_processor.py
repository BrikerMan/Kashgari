# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_class_processor.py
# time: 12:04 下午


import unittest
from tests.test_macros import TestMacros

from kashgari.generators import CorpusGenerator
from kashgari.processors.class_processor import ClassificationProcessor


class TestClassificationProcessor(unittest.TestCase):
    def test_processor(self):
        x_set, y_set = TestMacros.load_labeling_corpus('custom_1')
        corpus_gen = CorpusGenerator(x_set, y_set)

    def test_multi_label_processor(self):
        from kashgari.corpus import JigsawToxicCommentCorpus
        file_path = '/Users/brikerman/Downloads/jigsaw-toxic-comment-classification-challenge/train.csv'
        corpus = JigsawToxicCommentCorpus(file_path)
        x_set, y_set = corpus.load_data()

        corpus_gen = CorpusGenerator(x_set, y_set)

        processor = ClassificationProcessor(multi_label=True)
        processor.build_vocab_dict_if_needs(corpus_gen)
        r = processor.numerize_samples(y_set[20:40])
        print(r)
        print(processor.vocab2idx)


if __name__ == "__main__":
    pass
