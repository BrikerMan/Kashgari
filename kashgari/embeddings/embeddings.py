# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: embedding
@time: 2019-01-20

"""
import json
import logging
import os
from typing import Dict, Any

import keras_bert
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding
from keras.models import Model
from keras.preprocessing import sequence

import kashgari.macros as k
from kashgari.type_hints import *
from kashgari.utils import helper

EMBEDDINGS_PATH = os.path.join(k.DATA_PATH, 'embedding')


class BaseEmbedding(object):
    base_dict = {
        k.PAD: 0,
        k.BOS: 1,
        k.EOS: 2,
        k.UNK: 3
    }

    special_tokens = {
        k.PAD: k.PAD,
        k.UNK: k.UNK,
        k.BOS: k.BOS,
        k.EOS: k.EOS
    }

    def __init__(self,
                 name_or_path: str,
                 sequence_length: int,
                 embedding_size: int = None,
                 **kwargs):
        """
        init a WordEmbedding
        :param name_or_path: model name as `sgns.weibo.bigram` or model path like '/home/brikerman/w2v.model
        :param sequence_length: length of max sequence, all embedding is shaped as (sequence_length, embedding_size)
        :param embedding_size: embedding vector size, only need to set when using a CustomEmbedding
        :param kwargs: kwargs to pass to the method, func: `BaseEmbedding.build`
        """
        self.embedding_type = 'base'
        self.name = name_or_path
        self.embedding_size = embedding_size
        self._sequence_length = sequence_length
        self.model_path = ''
        self._token2idx: Dict[str, int] = None
        self._idx2token: Dict[int, str] = None
        self._model: Model = None
        self._kwargs = kwargs
        self.build(**kwargs)

    def update(self, info: Dict[str, Any]):
        self.name = info['name']
        self.embedding_type = info['embedding_type']
        self.embedding_size = info['embedding_size']
        self._sequence_length = info['sequence_length']
        self.model_path = info['model_path']
        self._kwargs = info['kwargs']

    def info(self):
        return {
            'embedding_type': self.embedding_type,
            'name': self.name,
            'embedding_size': self.embedding_size,
            'sequence_length': self._sequence_length,
            'model_path': self.model_path,
            'kwargs': self._kwargs
        }

    @property
    def token_count(self):
        return len(self._token2idx)

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, val):
        self._sequence_length = val
        self.build(**self._kwargs)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def token2idx(self):
        return self._token2idx

    @property
    def is_bert(self):
        return self.embedding_type == 'bert'

    @token2idx.setter
    def token2idx(self, value):
        self._token2idx = value
        self._idx2token = dict([(value, key) for (key, value) in value.items()])

    @property
    def idx2token(self):
        return self._idx2token

    def build(self, **kwargs):
        raise NotImplementedError()

    def build_token2idx_dict(self, x_data: List[TextSeqType], min_count: int = 5):
        raise NotImplementedError()

    def tokenize(self,
                 sentence: TextSeqInputType,
                 add_bos_eos: bool = True) -> TokenSeqInputType:
        is_list = isinstance(sentence[0], list)

        def tokenize_sentence(text: TextSeqType) -> TokenSeqType:
            tokens = [self.token2idx.get(token, self.token2idx[k.UNK]) for token in text]
            if add_bos_eos:
                tokens = [self.token2idx[k.BOS]] + tokens + [self.token2idx[k.BOS]]
            return tokens

        if is_list:
            return [tokenize_sentence(sen) for sen in sentence]
        else:
            return tokenize_sentence(sentence)

    def embed(self, sentence: TextSeqInputType) -> np.array:
        is_list = isinstance(sentence[0], list)
        tokens = self.tokenize(sentence)

        if is_list:
            embed_input = sequence.pad_sequences(tokens, self.sequence_length, padding='post')
        else:
            embed_input = sequence.pad_sequences([tokens], self.sequence_length, padding='post')

        embed_input = self.prepare_model_input(embed_input)
        print(embed_input)
        embed_pred = self.model.predict(embed_input)
        if is_list:
            return embed_pred
        else:
            return embed_pred[0]

    def prepare_model_input(self, input_x: np.array, **kwargs) -> np.array:
        return input_x


class WordEmbeddings(BaseEmbedding):
    base_dict = {
        k.PAD: 0,
        k.BOS: 1,
        k.EOS: 2,
        k.UNK: 3
    }

    URL_MAP = {
        'sgns.renmin.bigram': 'embedding/word2vec/sgns.renmin.bigram.bz2',
        'sgns.renmin.bigram-char': 'embedding/word2vec/sgns.renmin.bigram-char.bz2',
        'sgns.weibo.bigram': 'embedding/word2vec/sgns.weibo.bigram.bz2',
        'sgns.weibo.bigram-char': 'embedding/word2vec/sgns.weibo.bigram-char.bz2',
    }

    def get_embedding_matrix(self) -> np.array:
        base_matrix = []

        file = os.path.join(k.DATA_PATH, 'w2v_embedding_{}.json'.format(self.embedding_size))
        if os.path.exists(file):
            base_matrix = json.load(open(file, 'r', encoding='utf-8'))
            base_matrix = [np.array(matrix) for matrix in base_matrix]
        else:
            for index, key in enumerate(k.MARKED_KEYS):
                if index != 0:
                    vector = np.random.uniform(-0.5, 0.5, self.embedding_size)
                else:
                    vector = np.zeros(self.embedding_size)
                base_matrix.append(vector)
            with open(file, 'w', encoding='utf-8') as f:
                f.write(json.dumps([list(item) for item in base_matrix]))

        matrix_list = base_matrix + list(self.keyed_vector.vectors)
        return np.array(matrix_list)

    def build(self, **kwargs):
        self.embedding_type = 'word2vec'
        if self.name in WordEmbeddings.URL_MAP:
            url = self.URL_MAP.get(self.name)
            self.name = self.name + '.bz2'
        else:
            url = None

        self.model_path = helper.cached_path(self.name,
                                             url,
                                             sub_folders=['embedding', 'word2vec'])

        self.keyed_vector: KeyedVectors = KeyedVectors.load_word2vec_format(self.model_path, **kwargs)
        self.embedding_size = self.keyed_vector.vector_size

        word2idx = self.base_dict.copy()
        for word in self.keyed_vector.index2entity:
            word2idx[word] = len(word2idx)
        self.token2idx = word2idx

        input_layer = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_matrix = self.get_embedding_matrix()

        current = Embedding(self.token_count,
                            self.embedding_size,
                            input_length=self.sequence_length,
                            weights=[embedding_matrix],
                            trainable=False)(input_layer)
        self._model = Model(input_layer, current)
        logging.debug('------------------------------------------------')
        logging.debug('Loaded gensim word2vec model')
        logging.debug('model        : {}'.format(self.model_path))
        logging.debug('word count   : {}'.format(len(self.keyed_vector.index2entity)))
        logging.debug('Top 50 word  : {}'.format(self.keyed_vector.index2entity[:50]))
        logging.debug('------------------------------------------------')

    def build_token2idx_dict(self, x_data: List[TextSeqType], min_count: int = 5):
        logging.debug("word2vec embedding no need to build token2idx with corpus")


class BERTEmbedding(BaseEmbedding):
    base_dict = {}
    special_tokens = {
        k.PAD: '[PAD]',
        k.UNK: '[UNK]',
        k.BOS: '[CLS]',
        k.EOS: '[SEP]',
    }

    model_key_map = {
        'bert-base-uncased': 'uncased_L-12_H-768_A-12',
        'bert-large-uncased': 'uncased_L-24_H-1024_A-16',
        'bert-base-cased': 'cased_L-12_H-768_A-12',
        'bert-large-cased': 'cased_L-24_H-1024_A-16',
        'bert-base-multilingual-cased': 'multi_cased_L-12_H-768_A-12',
        'bert-base-chinese': 'chinese_L-12_H-768_A-12'
    }

    pre_trained_models = {
        # BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
        'uncased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                   'uncased_L-12_H-768_A-12.zip',
        # BERT-Large, Uncased
        # 24-layer, 1024-hidden, 16-heads, 340M parameters
        'uncased_L-24_H-1024_A-16': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                    'uncased_L-24_H-1024_A-16.zip',
        # BERT-Base, Cased
        # 12-layer, 768-hidden, 12-heads , 110M parameters
        'cased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                 'cased_L-12_H-768_A-12.zip',
        # BERT-Large, Cased
        # 24-layer, 1024-hidden, 16-heads, 340M parameters
        'cased_L-24_H-1024_A-16': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                  'cased_L-24_H-1024_A-16.zip',
        # BERT-Base, Multilingual Cased (New, recommended)
        # 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
        'multi_cased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_23/'
                                       'multi_cased_L-12_H-768_A-12.zip',
        # BERT-Base, Multilingual Uncased (Orig, not recommended)
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        'multilingual_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_03/'
                                        'multilingual_L-12_H-768_A-12.zip',
        # BERT-Base, Chinese
        # Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
        'chinese_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_03/'
                                   'chinese_L-12_H-768_A-12.zip',
    }

    def build(self):
        self.embedding_type = 'bert'
        url = self.pre_trained_models.get(self.model_key_map.get(self.name, self.name))
        self.model_path = helper.cached_path(self.model_key_map.get(self.name, self.name),
                                             url,
                                             ['embedding', 'bert'])

        config_path = os.path.join(self.model_path, 'bert_config.json')
        check_point_path = os.path.join(self.model_path, 'bert_model.ckpt')
        logging.info('loading bert model from {}\n'.format(self.model_path))
        model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                              check_point_path,
                                                              seq_len=self.sequence_length)
        output_layer = helper.NonMaskingLayer()(model.output)
        self._model = Model(model.inputs, output_layer)

        self.embedding_size = self.model.output_shape[-1]
        dict_path = os.path.join(self.model_path, 'vocab.txt')
        word2idx = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        for word in words:
            word2idx[word] = len(word2idx)
        for key, value in self.special_tokens.items():
            word2idx[key] = word2idx[value]

        self.token2idx = word2idx

    def build_token2idx_dict(self, x_data: List[TextSeqType], min_count: int = 5):
        logging.debug("bert embedding no need to build token2idx with corpus")

    def prepare_model_input(self, input_x: np.array, **kwargs) -> np.array:
        input_seg = np.zeros(input_x.shape)
        return [input_x, input_seg]


class CustomEmbedding(BaseEmbedding):

    def build(self, **kwargs):
        if self._token2idx is None:
            logging.debug('need to build after build_word2idx')
        else:
            input_x = Input(shape=(self.sequence_length,), dtype='int32')
            current = Embedding(self.token_count,
                                self.embedding_size)(input_x)
            self._model = Model(input_x, current)

    def build_token2idx_dict(self, x_data: List[TextSeqType], min_count: int = 5):
        if self.token2idx is None:
            word_set: Dict[str, int] = {}
            for x_item in x_data:
                for word in x_item:
                    word_set[word] = word_set.get(word, 0) + 1

            word2idx_list = sorted(word_set.items(), key=lambda kv: -kv[1])

            word2idx = self.base_dict.copy()
            for word, count in word2idx_list:
                if count >= min_count:
                    word2idx[word] = len(word2idx)

            self.token2idx = word2idx
        self.build()


if __name__ == '__main__':
    from kashgari.utils.logger import init_logger

    init_logger()
    embedding = WordEmbeddings('sgns.weibo.bigram.bz2', 10)

    sentence = '我 想 去 看 电影www'.split(' ')
    print(embedding.__dict__)

    print(embedding.tokenize(sentence))
    print(embedding.tokenize([sentence]))
    print(embedding.embed([sentence]))
