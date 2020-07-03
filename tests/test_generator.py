# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_generator.py
# time: 5:46 下午

import unittest

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.generators import CorpusGenerator, BatchDataSet
from kashgari.processors import SequenceProcessor
from tests.test_macros import TestMacros


class TestGenerator(unittest.TestCase):
    def test_corpus_generator(self):
        x_set, y_set = TestMacros.load_labeling_corpus()
        corpus_gen = CorpusGenerator(x_set, y_set)
        pass

    def test_batch_generator(self):
        x, y = ChineseDailyNerCorpus.load_data('valid')

        text_processor = SequenceProcessor()
        label_processor = SequenceProcessor(build_vocab_from_labels=True, min_count=1)

        corpus_gen = CorpusGenerator(x, y)

        text_processor.build_vocab_generator([corpus_gen])
        label_processor.build_vocab_generator([corpus_gen])

        batch_dataset1 = BatchDataSet(corpus_gen,
                                      text_processor=text_processor,
                                      label_processor=label_processor,
                                      segment=False,
                                      seq_length=None,
                                      max_position=100,
                                      batch_size=12)

        assert len(list(batch_dataset1.take())) == len(batch_dataset1)
        duplicate_len = len(batch_dataset1) * 2
        assert len(list(batch_dataset1.take(duplicate_len))) == duplicate_len
        assert len(list(batch_dataset1.take(1))) == 1

        for x, y in batch_dataset1.take(1):
            assert x.shape == y.shape == (12, 100)

        batch_dataset2 = BatchDataSet(corpus_gen,
                                      text_processor=text_processor,
                                      label_processor=label_processor,
                                      segment=False,
                                      seq_length=60,
                                      max_position=100,
                                      batch_size=12)

        for x, y in batch_dataset2.take(1):
            assert x.shape == y.shape == (12, 60)

        batch_dataset3 = BatchDataSet(corpus_gen,
                                      text_processor=text_processor,
                                      label_processor=label_processor,
                                      segment=False,
                                      seq_length=300,
                                      max_position=100,
                                      batch_size=12)

        for x, y in batch_dataset3.take(1):
            assert x.shape == y.shape == (12, 100)


if __name__ == '__main__':
    unittest.main()
