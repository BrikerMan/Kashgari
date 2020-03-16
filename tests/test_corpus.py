# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_corpus.py
# time: 10:47 上午

import unittest
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import SMP2018ECDTCorpus


class TestChineseDailyNerCorpus(unittest.TestCase):

    def test_load_data(self):
        train_x, train_y = ChineseDailyNerCorpus.load_data()
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0
        assert train_x[:5] != train_y[:5]

        test_x, test_y = ChineseDailyNerCorpus.load_data('test')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0

        test_x, test_y = ChineseDailyNerCorpus.load_data('valid')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0


class TestSMP2018ECDTCorpus(unittest.TestCase):

    def test_load_data(self):
        train_x, train_y = SMP2018ECDTCorpus.load_data()
        assert len(train_x) == len(train_y)
        assert len(train_x) > 0
        assert train_x[:5] != train_y[:5]

        test_x, test_y = SMP2018ECDTCorpus.load_data('test')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0

        test_x, test_y = SMP2018ECDTCorpus.load_data('valid')
        assert len(test_x) == len(test_y)
        assert len(test_x) > 0


if __name__ == "__main__":
    pass
