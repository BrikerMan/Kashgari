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
import os
import shutil
import unittest
import pytest
from kashgari.corpus import CoNLL2003Corpus, SMP2017ECDTClassificationCorpus
from kashgari.corpus import TencentDingdangSLUCorpus, ChinaPeoplesDailyNerCorpus
from kashgari.macros import DATA_PATH
from kashgari.utils.logger import init_logger

init_logger()


class TestTencentDingdangSLUCorpus(object):

    def test_tencent_dingdang(self):
        x, y = TencentDingdangSLUCorpus.get_classification_data()
        assert len(x) > 0
        assert len(x) == len(y)

    def test_download(self):
        shutil.rmtree(os.path.join(DATA_PATH, 'corpus'), ignore_errors=True)
        self.test_tencent_dingdang()


class TestChinaPeoplesDailyNerCorpus(object):

    def test_tagging_data(self):
        x, y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
        assert len(x) > 0
        assert len(x) == len(y)

    def test_download(self):
        shutil.rmtree(os.path.join(DATA_PATH, 'corpus'), ignore_errors=True)
        self.test_tagging_data()


class TestCoNLL2003Corpus(object):

    def test_tagging_data(self):
        x, y = CoNLL2003Corpus.get_sequence_tagging_data()
        assert len(x) > 0
        assert len(x) == len(y)

    def test_download(self):
        shutil.rmtree(os.path.join(DATA_PATH, 'corpus'), ignore_errors=True)
        self.test_tagging_data()


class TestSMP2017ECDTClassificationCorpus(object):

    def test_tagging_data(self):
        x, y = SMP2017ECDTClassificationCorpus.get_classification_data()
        assert len(x) > 0
        assert len(x) == len(y)

    def test_download(self):
        shutil.rmtree(os.path.join(DATA_PATH, 'corpus'), ignore_errors=True)
        self.test_tagging_data()


if __name__ == "__main__":
    pass
