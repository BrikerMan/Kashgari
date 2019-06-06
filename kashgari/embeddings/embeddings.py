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
from itertools import chain
from collections import Counter

import keras_bert
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding, concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, BytePairEncoding

import kashgari.macros as k
from kashgari.type_hints import *
from kashgari.utils import helper
from kashgari.layers import NonMaskingLayer



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
                 sequence_length: int = None,
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
                tokens = [self.token2idx[k.BOS]] + tokens + [self.token2idx[k.EOS]]
            return tokens

        if is_list:
            return [tokenize_sentence(sen) for sen in sentence]
        else:
            return tokenize_sentence(sentence)

    def embed(self, sentence: TextSeqInputType, seq_idx: int=0) -> np.array:
        is_list = isinstance(sentence[0], list)
        tokens = self.tokenize(sentence)

        if not is_list:
            tokens = [tokens]
        if isinstance(self.sequence_length, int):
            embed_input = sequence.pad_sequences(tokens, self.sequence_length, padding='post')
        elif isinstance(self.sequence_length, list):
            embed_input = sequence.pad_sequences(tokens, self.sequence_length[seq_idx], padding='post')

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


    def __init__(self,
                 name_or_path: str,
                 sequence_length: int = None,
                 embedding_size: int = None,
                 layer_nums: int = 4,
                 trainable: bool = False,
                 **kwargs,):
        """
        init a WordEmbedding
        :param name_or_path: model name as `sgns.weibo.bigram` or model path like '/home/brikerman/w2v.model
        :param sequence_length: length of max sequence, all embedding is shaped as (sequence_length, embedding_size)
        :param embedding_size: embedding vector size, only need to set when using a CustomEmbedding
        :param layer_nums: number of layers whose outputs will be concatenated as a single output.
                           default `4`, the last 4 hidden layers
        :param trainable: whether if the output feature layer is trainable, default `False` and set it to `True` for finetune
        :param kwargs: kwargs to pass to the method, func: `BaseEmbedding.build`
        """
        self.layer_nums = layer_nums
        self.trainable = trainable
        self.training = False # We do not need to train the whole bert model so set it to `False`
        super(BERTEmbedding, self).__init__(name_or_path, sequence_length, embedding_size, **kwargs)


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
                                                              seq_len=self.sequence_length,
                                                              output_layer_num=self.layer_nums,
                                                              training=self.training,
                                                              trainable=self.trainable
                                                              )
        #num_layers = len(model.layers)
        #features_layers = [model.get_layer(index=num_layers-1+idx*8).output\
        #                    for idx in range(-3, 1)]
        #embedding_layer = concatenate(features_layers)
        #output_layer = NonMaskingLayer()(embedding_layer)
        output_layer = NonMaskingLayer()(model.output)
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
    def __init__(self,
                 name_or_path: str = 'custom-embedding',
                 sequence_length: int = None,
                 embedding_size: int = None,
                 **kwargs):
        """
        :param name_or_path: just a name for custom embedding
        :param sequence_length: length of max sequence, all embedding is shaped as (sequence_length, embedding_size)
        :param embedding_size: embedding vector size, only need to set when using a CustomEmbedding
        :param kwargs: kwargs to pass to the method, func: `BaseEmbedding.build`
        """
        if sequence_length is None or embedding_size is None:
            raise ValueError('Must set sequence_length and sequence_length when using the CustomEmbedding layer')
        super(CustomEmbedding, self).__init__(name_or_path, sequence_length, embedding_size, **kwargs)

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
            #word_set: Dict[str, int] = {}
            # for x_item in x_data:
            #     for word in x_item:
            #         word_set[word] = word_set.get(word, 0) + 1
            data_depth = helper.depth_count(x_data)
            if data_depth > 1:
                x_items = x_data
                for _ in range(data_depth-1):
                    x_items = list(chain(*x_items))
            word_freq = Counter(x_items)
            # word_set = {word: freq for word, freq in word_freq.items() if freq >= min_count}
            # word2idx_list = sorted(word_set.items(), key=lambda kv: -kv[1])
            word2idx_list = sorted(word_freq.items(), key=lambda kv: -kv[1])

            word2idx = self.base_dict.copy()
            offset = len(word2idx)
            # for word, count in word2idx_list:
            #     if count >= min_count:
            #         word2idx[word] = len(word2idx)
            for idx, (word, freq) in enumerate(word2idx_list):
                if freq >= min_count:
                    word2idx[word] = idx + offset

            self.token2idx = word2idx
        self.build()


class TwoHeadEmbedding(CustomEmbedding):
    def __init__(self,
                 name_or_path: str = 'twohead-embedding',
                 sequence_length: List[int] = None,
                 embedding_size: int = None,
                 **kwargs):
        """
        Inheritated from CustomEmbedding class.
        :param name_or_path: just a name for two head embedding
        :param sequence_length: max length list of sequences, all embedding is shaped as (sequence_length[idx], embedding_size)
        :param embedding_size: embedding vector size, only need to set when using a CustomEmbedding or its subclass
        :param kwargs: kwargs to pass to the method, func: `BaseEmbedding.build`
        """
        if sequence_length is None or embedding_size is None:
            raise ValueError('Must set all sequence_length and embedding_size when using the TwoheadEmbedding layer')
        super(TwoHeadEmbedding, self).__init__(name_or_path, sequence_length, embedding_size, **kwargs)

    def build(self, **kwargs):
        self.embedding_type = 'twohead'
        if self._token2idx is None:
            logging.debug('need to build after build_word2idx')
        else:
            input_x1 = Input(shape=(self.sequence_length[0],), dtype='int32', name='master_input')
            current1 = Embedding(self.token_count,
                                self.embedding_size)(input_x1)
            input_x2 = Input(shape=(self.sequence_length[1],), dtype='int32', name='assist_input')
            current2 = Embedding(self.token_count,
                                self.embedding_size)(input_x2)
            current = concatenate([current1, current2], axis=1)
            self._model = Model(inputs=[input_x1, input_x2], outputs=current)

    def build_token2idx_dict(self, x_data: List[TextSeqType], min_count: int = 5):
        super(TwoHeadEmbedding, self).build_token2idx_dict(x_data, min_count)

    def embed(self, sentences_pair: List[List[TextSeqInputType]]) -> np.array:
        embed_inputs = []
        for idx, sentences in enumerate(sentences_pair):
            is_list = isinstance(sentences[0], list)
            tokens = self.tokenize(sentences)
            if not is_list:
                tokens = [tokens]
            if isinstance(self.sequence_length, list):
                embed_input = sequence.pad_sequences(tokens, self.sequence_length[idx], padding='post')
            elif isinstance(self.sequence_length, int):
                embed_input = sequence.pad_sequences(tokens, self.sequence_length, padding='post')
            embed_inputs.append(embed_input)
        embed_inputs = self.prepare_model_input(embed_inputs)
        print(embed_inputs)
        embed_pred = self.model.predict(embed_inputs)
        return embed_pred


class GPT2Embedding(BaseEmbedding):

    def build(self, **kwargs):
        self.embedding_type = 'gpt2'

        config_path = os.path.join(self.name, 'hparams.json')
        checkpoint_path = os.path.join(self.name, 'model.ckpt')
        encoder_path = os.path.join(self.name, 'encoder.json')
        vocab_path = os.path.join(self.name, 'vocab.bpe')

        self._model: Model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        for layer in self._model.layers:
            layer.trainable = False

        self.bpe: BytePairEncoding = get_bpe_from_files(encoder_path, vocab_path)

        word2idx = self.bpe.token_dict.copy()
        word2idx[k.PAD] = word2idx['pad']
        word2idx[k.UNK] = word2idx['unk']
        word2idx[k.BOS] = word2idx['pad']
        word2idx[k.EOS] = word2idx['pad']
        self.token2idx = word2idx

    def build_token2idx_dict(self, x_data: List[TextSeqType], min_count: int = 5):
        logging.debug("word2vec embedding no need to build token2idx with corpus")


if __name__ == '__main__':
    train_x = [
        list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
        list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
        list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
        list('语言学包含了几种分支领域。'),
        list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
    ]
    train_y = ['a', 'a', 'a', 'b', 'c']

    from kashgari.utils.logger import init_logger
    from kashgari.tasks.classification import CNNModel
    init_logger()
    embedding = GPT2Embedding('/Users/brikerman/Desktop/python/gpt-2/models/117M', 10)
    r = embedding.embed(['hello', 'world'])
    model = CNNModel(embedding)
    model.fit(train_x, train_y, epochs=20)
    print(r.shape)
