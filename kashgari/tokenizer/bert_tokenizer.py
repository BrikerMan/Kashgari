# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bert_tokenizer.py
# time: 11:33 上午

# flake8: noqa: E127

import codecs
import os
import unicodedata

from kashgari.tokenizer.base_tokenizer import Tokenizer

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


class BertTokenizer(Tokenizer):

    """
    Bert Like Tokenizer, ref: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

    """
    def __init__(self,
                 token_dict=None,
                 token_cls=TOKEN_CLS,
                 token_sep=TOKEN_SEP,
                 token_unk=TOKEN_UNK,
                 pad_index=0,
                 cased=False):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        self._token_dict = token_dict
        if self._token_dict:
            self._token_dict_inv = {v: k for k, v in token_dict.items()}
        else:
            self._token_dict_inv = {}
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._pad_index = pad_index
        self._cased = cased

    @classmethod
    def load_from_model(cls, model_path: str):
        dict_path = os.path.join(model_path, 'vocab.txt')

        token2idx = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token2idx[token] = len(token2idx)
        return BertTokenizer(token_dict=token2idx)

    @classmethod
    def load_from_vacab_file(cls, vacab_path: str):
        token2idx = {}
        with codecs.open(vacab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token2idx[token] = len(token2idx)
        return BertTokenizer(token_dict=token2idx)

    def tokenize(self, first):
        """Split text to tokens.

        :param first: First text.
        :param second: Second text.

        :return: A list of strings.
        """
        tokens = self._tokenize(first)
        return tokens

    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        if self._token_dict:
            tokens = []
            for word in spaced.strip().split():
                tokens += self._word_piece_tokenize(word)
            return tokens
        else:
            return spaced.strip().split()

    def _word_piece_tokenize(self, word):
        if word in self._token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens

    @staticmethod
    def _is_punctuation(ch):  # noqa: E127
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch):
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')
