#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/4 6:46 上午
# File    : test_tokenizers.py
# Project : Kashgari

import unittest
import numpy as np
import os
from kashgari.tokenizers import Tokenizer, JiebaTokenizer, BertTokenizer
from tests.test_macros import TestMacros


class TestUtils(unittest.TestCase):

    def test_jieba_tokenizer(self):
        os.system("pip3 uninstall -y jieba")

        with self.assertRaises(ModuleNotFoundError):
            _ = JiebaTokenizer()

        os.system("pip3 install  jieba")
        t = JiebaTokenizer()
        assert ['你好', '世界', '!', ' ', 'Hello', ' ', 'World'] == t.tokenize('你好世界! Hello World')

    def test_base_tokenizer(self):
        t = Tokenizer()
        assert ['Hello', 'World'] == t.tokenize('Hello World')

    def test_bert_tokenizer(self):
        bert_path = TestMacros.bert_path
        vocab_path = os.path.join(bert_path, 'vocab.txt')
        tokenizer = BertTokenizer.load_from_vocab_file(vocab_path)

        assert ['你', '好', '世', '界', '!',
                'h', '##e', '##l', '##l', '##o',
                'w', '##o', '##r', '##l', '##d'] == tokenizer.tokenize("你好世界! Hello World")
        assert ['jack', 'makes', 'c', '##a', '##k', '##e'] == tokenizer.tokenize("Jack makes cake")
        assert ['你', '好', '呀'] == tokenizer.tokenize("你好呀")

        tokenizer = BertTokenizer()
        assert ['你', '好', '世', '界', '!', 'hello', 'world'] == tokenizer.tokenize("你好世界! Hello World")
        assert ['jack', 'makes', 'cake'] == tokenizer.tokenize("Jack makes cake")
        assert ['你', '好', '呀'] == tokenizer.tokenize("你好呀")
