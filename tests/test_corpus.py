# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: test_corpus.py
@time: 2019-01-31 13:56

"""
import unittest

from kashgari.corpus import TencentDingdangSLUCorpus
from kashgari.corpus import ChinaPeoplesDailyNerCorpus
from kashgari.corpus import CoNLL2003Corpus
from kashgari.corpus import SMP2017ECDTClassificationCorpus


class TestTencentDingdangSLUCorpus(unittest.TestCase):
    def test_get_classification_data(self):
        train_x, train_y = TencentDingdangSLUCorpus.get_classification_data('train')
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0

        test_x, test_y = TencentDingdangSLUCorpus.get_classification_data('test')
        assert len(test_x) == len(test_y)

    def test_get_sequence_tagging_data(self):
        train_x, train_y = TencentDingdangSLUCorpus.get_sequence_tagging_data(is_test=False)
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0


class TestChinaPeoplesDailyNerCorpus(unittest.TestCase):
    def test_ner_data(self):
        train_x, train_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data('train')
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0

        test_x, test_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data('test')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0


class TestCoNLL2003Corpus(unittest.TestCase):
    def test_ner_data(self):
        train_x, train_y = CoNLL2003Corpus.get_sequence_tagging_data('train')
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0

        test_x, test_y = CoNLL2003Corpus.get_sequence_tagging_data('test')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0


class TestSMP2017ECDTClassificationCorpus(unittest.TestCase):
    def test_ner_data(self):
        train_x, train_y = SMP2017ECDTClassificationCorpus.get_classification_data('train')
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0

        test_x, test_y = SMP2017ECDTClassificationCorpus.get_classification_data('train')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0


if __name__ == "__main__":
    unittest.main()
