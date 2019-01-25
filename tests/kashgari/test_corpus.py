# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: test_corpus.py
@time: 2019-01-25 18:01

"""
import shutil
import unittest

from kashgari.macros import DATA_PATH
from kashgari.corpus import TencentDingdangSLUCorpus, ChinaPeoplesDailyNerCorpus
from kashgari.corpus import CoNLL2003Corpus, SMP2017ECDTClassificationCorpus

from kashgari.utils.logger import init_logger
init_logger()


class TencentDingdangSLUCorpusTest(unittest.TestCase):

    def test_tencent_dingdang(self):
        x, y = TencentDingdangSLUCorpus.get_classification_data()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))

    def test_download(self):
        shutil.rmtree(DATA_PATH, ignore_errors=True)
        self.test_tencent_dingdang()


class ChinaPeoplesDailyNerCorpusTest(unittest.TestCase):

    def test_tagging_data(self):
        x, y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))

    def test_download(self):
        shutil.rmtree(DATA_PATH, ignore_errors=True)
        self.test_tagging_data()


class CoNLL2003CorpusTest(unittest.TestCase):

    def test_tagging_data(self):
        x, y = CoNLL2003Corpus.get_sequence_tagging_data()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))

    def test_download(self):
        shutil.rmtree(DATA_PATH, ignore_errors=True)
        self.test_tagging_data()


class SMP2017ECDTClassificationCorpusTest(unittest.TestCase):

    def test_tagging_data(self):
        x, y = SMP2017ECDTClassificationCorpus.get_classification_data()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))

    def test_download(self):
        shutil.rmtree(DATA_PATH, ignore_errors=True)
        self.test_tagging_data()


if __name__ == "__main__":
    unittest.main()
