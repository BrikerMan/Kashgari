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
import logging
import numpy as np
from typing import List, Union, Dict

from kashgari.embedding.word2vec import Word2Vec
from kashgari.macros import PAD, BOS, EOS, UNK
from kashgari import k


class Tokenizer(object):
    def __init__(self,
                 embedding_name: Union[k.Word2VecModels, str],
                 sequence_length: int = None,
                 segmenter: k.SegmenterType = k.SegmenterType.space,
                 **kwargs):
        self.word2idx = {
            PAD: 0,
            BOS: 1,
            EOS: 2,
            UNK: 3
        }
        self.idx2word = {
            PAD: 0
        }

        self.label2idx = {}
        self.idx2label = {}

        self.sequence_length = sequence_length
        self.embedding_name = embedding_name
        self.embedding_size = 0
        self.embedding_limit = kwargs.get('embedding_limit', 10000)

        self.segmenter = segmenter
        self.kwargs = kwargs

        self.embedding = None

        self.load_word2vec(limit=self.embedding_limit, **kwargs)

    @classmethod
    def get_recommend_tokenizer(cls):
        return Tokenizer(embedding_name=k.Word2VecModels.sgns_weibo_bigram,
                         sequence_length=80,
                         segmenter=k.SegmenterType.jieba)

    @property
    def class_num(self) -> int:
        return len(self.label2idx)

    def build_with_corpus(self,
                          x_data: Union[List[List[str]], List[str]],
                          y_data: Union[List[List[str]], List[str]],
                          **kwargs):
        label_set: Dict[str, int] = {}
        for y_item in y_data:
            if isinstance(y_item, list):
                for y in y_item:
                    label_set[y] = label_set.get(y, 0) + 1
            else:
                label_set[y_item] = label_set.get(y_item, 0) + 1

        label2idx = {}
        for label in label_set.keys():
            label2idx[label] = len(label2idx)
        idx2label = dict([(value, key) for (key, value) in label2idx.items()])

        self.label2idx = label2idx
        self.idx2label = idx2label

        logging.info('----- build label2index map finished ------')
        for label, count in label_set.items():
            logging.info('{:10}: {} items'.format(label, count))
        logging.info('{:10}: {}'.format('label2idx', self.label2idx))
        logging.info('-------------------------------------------')

    def get_embedding_matrix(self, **kwargs) -> np.array:
        # w2v: Word2Vec = Word2Vec(self.embedding, **kwargs)
        return self.embedding.get_embedding_matrix()

    def load_word2vec(self, **kwargs):
        """
        load word2vec embedding with gensim
        """
        embedding: Word2Vec = Word2Vec(self.embedding_name, **kwargs)
        self.embedding = embedding
        for word in embedding.keyed_vector.index2entity:
            self.word2idx[word] = len(self.word2idx)

        self.embedding_size = embedding.embedding_size
        self.__create_reversed_dict__()

    def __create_reversed_dict__(self):
        self.idx2word = dict([(value, key) for (key, value) in self.word2idx.items()])

    def word_to_token(self,
                      sentence: Union[List[str], str],
                      add_prefix_suffix: bool = True,
                      **kwargs) -> List[int]:
        """
        convert sentence to tokens
        :param sentence: sentence ['我', '想', '你'] or '我 想 你'
        :param add_prefix_suffix: 是否添加前后缀
        :param kwargs:
        :return:
        """
        if isinstance(sentence, str):
            sentence = self.segment(sentence)
        tokens = [self.word2idx.get(word, self.word2idx[UNK]) for word in sentence]
        if add_prefix_suffix:
            tokens = [self.word2idx[BOS]] + tokens + [self.word2idx[EOS]]
        return tokens

    def token_to_word(self,
                      tokens: List[int],
                      sequence_length: int = 0,
                      remove_bos_eos: bool = True,
                      **kwargs):
        words = [self.idx2word[token] for token in tokens]
        if remove_bos_eos:
            if words[0] == BOS:
                words = words[1:]
            if words[-1] == EOS:
                words = words[:-1]

        if sequence_length:
            words = words[:sequence_length]
        return words

    def label_to_token(self, label: str) -> int:
        return self.label2idx[label]

    def token_to_label(self, token: int) -> str:
        return self.idx2label[token]

    def segment(self, text: str) -> List[str]:
        text = text.strip()
        if self.segmenter == k.SegmenterType.jieba:
            import jieba
            return list(jieba.cut(text))
        elif self.segmenter == k.SegmenterType.space:
            return text.split(' ')
        else:
            return list(text)


if __name__ == '__main__':
    from kashgari.utils.logger import init_logger
    init_logger()
    t = Tokenizer(k.Word2VecModels.sgns_weibo_bigram)
    t.segmenter = k.SegmenterType.char
    t_tokens = t.word_to_token('今天天气不错')
    print(t_tokens)
    t_words = t.token_to_word(t_tokens, remove_bos_eos=False)
    print(t_words)
