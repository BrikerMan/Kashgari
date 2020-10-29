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

        batch_dataset = BatchDataSet(corpus_gen,
                                     text_processor=text_processor,
                                     label_processor=label_processor,
                                     segment=False,
                                     seq_length=None,
                                     max_position=100,
                                     batch_size=12)

        duplicate_len = len(batch_dataset)
        assert len(list(batch_dataset.take(duplicate_len))) == duplicate_len
        assert len(list(batch_dataset.take(1))) == 1

    def test_huge_batch_size(self):
        x, y = [['this', 'is', 'Jack', 'Ma']], [['O', 'O', 'B', 'I']]

        text_processor = SequenceProcessor()
        label_processor = SequenceProcessor(build_vocab_from_labels=True, min_count=1)

        corpus_gen = CorpusGenerator(x, y)

        text_processor.build_vocab_generator([corpus_gen])
        label_processor.build_vocab_generator([corpus_gen])

        batch_dataset = BatchDataSet(corpus_gen,
                                     text_processor=text_processor,
                                     label_processor=label_processor,
                                     segment=False,
                                     seq_length=None,
                                     max_position=100,
                                     batch_size=512)

        for x_b, y_b in batch_dataset.take(1):
            print(y_b.shape)
        duplicate_len = len(batch_dataset)
        assert len(list(batch_dataset.take(duplicate_len))) == duplicate_len
        assert len(list(batch_dataset.take(1))) == 1


if __name__ == '__main__':
    unittest.main()
