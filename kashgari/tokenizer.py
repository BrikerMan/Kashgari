# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: tokenizer.py
@time: 2019-01-19 09:57

"""
from typing import List, Union

import numpy as np

from kashgari.embedding.word2vec import Word2Vec
from kashgari.macros import PAD, BOS, EOS, UNK, NO_TAG
from kashgari.utils import k


class Tokenizer(object):
    def __init__(self, embedding: Union[k.Word2VecModels, str], **kwargs):
        self.word2idx = {
            PAD: 0,
            BOS: 1,
            EOS: 2,
            UNK: 3
        }
        self.idx2word = {
            PAD: 0,
            NO_TAG: 1
        }

        self.label2idx = {}
        self.idx2label = {}

        self.sequence_length = 30
        self.embedding_size = 100

        self.data = {}

        self.embedding = embedding

        self.kwargs = kwargs
        self.w2v = None
        self._build_(limit=10000)

    def _build_(self, **kwargs):
        self.load_word2vec(**kwargs)

    def get_embedding_matrix(self, **kwargs) -> np.array:
        # w2v: Word2Vec = Word2Vec(self.embedding, **kwargs)
        return self.w2v.get_embedding_matrix()

    def load_word2vec(self, **kwargs):
        """
        load word2vec embedding with gensim
        """
        w2v: Word2Vec = Word2Vec(self.embedding, **kwargs)
        self.w2v = w2v
        for word in w2v.keyed_vector.index2entity:
            self.word2idx[word] = len(self.word2idx)

        self.embedding_size = w2v.embedding_size
        self.__create_reversed_dict__()

    def __create_reversed_dict__(self):
        self.idx2word = dict([(value, key) for (key, value) in self.word2idx.items()])

    def word_to_token(self,
                      sentence: Union[List[str], str],
                      add_prefix_suffix: bool = False,
                      **kwargs) -> List[int]:
        """
        convert sentence to tokens
        :param sentence: sentence ['我', '想', '你'] or '我 想 你'
        :param add_prefix_suffix: 是否添加前后缀
        :param kwargs:
        :return:
        """
        tokens = [self.word2idx.get(word, self.word2idx[UNK]) for word in sentence]
        if add_prefix_suffix:
            tokens = [self.word2idx[BOS]] + tokens + [self.word2idx[EOS]]
        return tokens

    def token_to_word(self,
                      tokens: List[int],
                      sequence_length: int = 0,
                      **kwargs):
        words = [self.idx2word[token] for token in tokens]

        if words[0] == BOS:
            words = words[1:]
        if words[-1] == EOS:
            words = words[:-1]

        if sequence_length:
            words = words[:sequence_length]
        return words

    def label_to_tokens(self):
        pass

    def tokens_to_label(self):
        pass


if __name__ == '__main__':
    from kashgari.utils.logger import init_logger
    init_logger()
    path = '/Users/brikerman/Downloads/sgns.weibo.bigram'
    t = Tokenizer(k.Word2VecModels.sgns_weibo_bigram, limit=10000)
    t_tokens = t.word_to_token(['风格', '不', '一样', '嘛', '，', '都', '喜欢', '！', '最', '喜欢', '哪张', '？'], add_prefix_suffix=True)
    print(t_tokens)
    t_words = t.token_to_word(t_tokens)
    print(t_words)
