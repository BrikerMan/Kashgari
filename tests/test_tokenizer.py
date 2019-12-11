# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_tokenizer.py
# time: 11:47 上午

import os
import unittest

from tensorflow.keras.utils import get_file

from kashgari.macros import DATA_PATH
from kashgari.tokenizer import BertTokenizer, JiebaTokenizer, Tokenizer

bert_path = get_file('bert_sample_model',
                     "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                     cache_dir=DATA_PATH,
                     untar=True)


class TestTokenizer(unittest.TestCase):
    def test_basic(self):
        t = Tokenizer()
        t.tokenize('hello 你好')
        t.tokenize("It's really amazing")


class TestBertTokenizer(unittest.TestCase):
    def test_basic(self):
        b0 = BertTokenizer()
        b0.tokenize('hello 你好')
        b0.tokenize("It's really amazing")
        b1 = BertTokenizer({'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, '[MASK]': 3, 'hello': 4})
        b1.tokenize('hello 你好')
        b1.tokenize("It's really amazing")
        b2 = BertTokenizer.load_from_model(bert_path)
        b2.tokenize("It's really amazing")


class TestJiebaTokenizer(unittest.TestCase):
    def test_basic(self):
        os.system('pip uninstall -y jieba')
        self.assertRaises(ModuleNotFoundError, JiebaTokenizer)

        os.system('pip install jieba')
        j = JiebaTokenizer()
        j.tokenize('你好，Kashgari')


if __name__ == "__main__":
    pass
