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
from typing import List, Union, Dict

from kashgari import k
from kashgari.embedding import EmbeddingModel, CustomEmbedding, BERTEmbedding
from kashgari.macros import PAD, BOS, EOS, UNK


class Tokenizer(object):
    def __init__(self,
                 embedding: EmbeddingModel = None,
                 sequence_length: int = None,
                 segmenter: k.SegmenterType = k.SegmenterType.jieba,
                 **kwargs):
        self.embedding = embedding

        self.word2idx = {}
        self.idx2word = {}

        self.idx2word = {
            PAD: 0
        }

        self.label2idx = {}
        self.idx2label = {}

        self.sequence_length = sequence_length

        self.segmenter = segmenter
        self.kwargs = kwargs
        if embedding is None:
            embedding = CustomEmbedding(embedding_size=100)
        self.embedding = embedding

        if not isinstance(self.embedding, CustomEmbedding):
            self.word2idx = self.embedding.get_word2idx_dict()
            self.idx2word = dict([(value, key) for (key, value) in self.idx2word.items()])

    @classmethod
    def get_recommend_tokenizer(cls):
        return Tokenizer(embedding_name=k.Word2VecModels.sgns_weibo_bigram,
                         sequence_length=80,
                         segmenter=k.SegmenterType.jieba)

    @property
    def class_num(self) -> int:
        return len(self.label2idx)

    @property
    def word_num(self) -> int:
        return len(self.word2idx)

    @property
    def is_bert(self) -> bool:
        if self.embedding is None:
            raise NotImplementedError('please set embedding for tokenize')
        return isinstance(self.embedding, BERTEmbedding)

    # noinspection PyTypeChecker,PyTypeChecker
    def build_with_corpus(self,
                          x_data: Union[List[List[str]], List[str]],
                          y_data: Union[List[List[str]], List[str]],
                          only_if_needs: bool = True,
                          **kwargs):
        if isinstance(self.embedding, CustomEmbedding):
            word_set: Dict[str, int] = {}
            for x_item in x_data:
                if isinstance(x_item, list):
                    for y in x_item:
                        for word in self.segment(y):
                            word_set[word] = word_set.get(word, 0) + 1
                elif isinstance(x_item, str):
                    for word in self.segment(x_item):
                        word_set[word] = word_set.get(word, 0) + 1

            word2idx_list = sorted(word_set.items(), key=lambda kv: -kv[1])

            word2idx = CustomEmbedding.base_dict
            for word, count in word2idx_list:
                if count >= 2:
                    word2idx[word] = len(word2idx)

            # word2idx = dict([(key, value) for (key, value) in word2idx_dict.items()])
            idx2word = dict([(value, key) for (key, value) in word2idx.items()])
            self.word2idx = word2idx
            self.idx2word = idx2word
        else:
            self.word2idx: Dict = self.embedding.get_word2idx_dict()
            self.idx2word = dict([(value, key) for (key, value) in self.idx2word.items()])

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
    t = Tokenizer()
    t.segmenter = k.SegmenterType.char
    t_tokens = t.word_to_token('今天天气不错')
    print(t_tokens)
    t_words = t.token_to_word(t_tokens, remove_bos_eos=False)
    print(t_words)
