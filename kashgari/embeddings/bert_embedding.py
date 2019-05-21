# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_embedding.py
# time: 2019-05-20 17:40

import logging
import os
import numpy as np
os.environ['TF_KERAS'] = '1'
import keras_bert

from typing import Union, Optional, Dict, Any, List
from tensorflow import keras

from kashgari.embeddings.base_embedding import Embedding
from kashgari.pre_processors import PreProcessor

L = keras.layers


class BertEmbedding(Embedding):
    """Pre-trained word2vec embedding"""

    def __init__(self,
                 bert_path: str,
                 sequence_length: Union[int, str] = 'auto',
                 processor: Optional[PreProcessor] = None,
                 **kwargs):
        """
        Args:
            w2v_path: word2vec file path
            w2v_kwargs: params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
        """
        super(BertEmbedding, self).__init__(sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor)

        self.processor.token_pad = '[PAD]'
        self.processor.token_unk = '[UNK]'
        self.processor.token_bos = '[CLS]'
        self.processor.token_eos = '[SEP]'

        self.bert_path = bert_path

    def _build_token2idx_from_bert(self):
        token2idx = {
            self.processor.token_pad: 0,
            self.processor.token_unk: 1,
            self.processor.token_bos: 2,
            self.processor.token_eos: 3
        }

        config_path = os.path.join(self.bert_path, 'bert_config.json')
        check_point_path = os.path.join(self.bert_path, 'bert_model.ckpt')
        self.bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                                        check_point_path,
                                                                        seq_len=12)

        dict_path = os.path.join(self.bert_path, 'vocab.txt')

        with open(dict_path, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        for idx, word in enumerate(words):
            token2idx[word] = idx
        self.bert_token2idx = token2idx

        # features_layers = [model.get_layer(index=num_layers - 1 + idx * 8).output \
        #                    for idx in range(-3, 1)]
        # embedding_layer = concatenate(features_layers)
        # output_layer = NonMaskingLayer()(embedding_layer)
        # # output_layer = NonMaskingLayer()(model.output)
        # self._model = Model(model.inputs, output_layer)
        #
        # self.embedding_size = self.model.output_shape[-1]
        # dict_path = os.path.join(self.model_path, 'vocab.txt')
        # word2idx = {}
        # with open(dict_path, 'r', encoding='utf-8') as f:
        #     words = f.read().splitlines()
        # for idx, word in enumerate(words):
        #     word2idx[word] = idx
        #     # word2idx[word] = len(word2idx)
        # for key, value in self.special_tokens.items():
        #     word2idx[key] = word2idx[value]

    def build_model(self, **kwargs):
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            # input_tensor = L.Input(shape=(self.sequence_length,),
            #                        name='inputs')
            # layer_embedding = L.Embedding(self.token_count,
            #                               self.embedding_size,
            #                               weights=[self.w2v_vector_matrix],
            #                               trainable=False,
            #                               name='layer_embedding')
            #
            # embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = self.bert_model

    def prepare_for_labeling(self,
                             x: List[List[str]],
                             y: List[List[str]]):
        if len(self.processor.token2idx) == 0:
            self._build_token2idx_from_bert()
            self.processor.token2idx = self.bert_token2idx

        print(list(self.bert_token2idx.items())[:105])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    b = BertEmbedding('/Users/brikerman/.kashgari/embedding/bert/chinese_L-12_H-768_A-12/',
                      sequence_length=12)

    from kashgari.corpus import SMP2018ECDTCorpus

    test_x, test_y = SMP2018ECDTCorpus.load_data('valid')

    b.prepare_for_labeling(test_x, test_y)
    print(b.embed(list('我想你啊啊啊啊啊啊啊啊')))
