# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_class_processor.py
# time: 12:04 下午


import unittest
from tests.test_macros import TestMacros

from kashgari.generators import CorpusGenerator
from kashgari.processors import ClassificationProcessor


class TestClassificationProcessor(unittest.TestCase):
    def test_processor(self):
        x_set, y_set = TestMacros.load_classification_corpus()
        processor = ClassificationProcessor()
        processor.build_vocab(x_set, y_set)
        r = processor.transform(y_set[20:40])
        print(r)
        print(processor.vocab2idx)

    def test_multi_label_processor(self):
        from kashgari.corpus import JigsawToxicCommentCorpus
        file_path = TestMacros.jigsaw_mini_corpus_path
        corpus = JigsawToxicCommentCorpus(file_path)
        x_set, y_set = corpus.load_data()

        corpus_gen = CorpusGenerator(x_set, y_set)

        processor = ClassificationProcessor(multi_label=True)
        processor.build_vocab_generator(corpus_gen)
        r = processor.transform(y_set[20:40])
        print(r)
        print(processor.vocab2idx)

        processor = ClassificationProcessor(multi_label=True)
        processor.build_vocab(x_set, y_set)
        r = processor.transform(y_set[20:40])
        print(r)
        print(processor.vocab2idx)


if __name__ == "__main__":
    pass
