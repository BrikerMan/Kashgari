# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: experimental.py
# time: 2019-05-22 19:35

import logging
from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.labeling.base_model import BaseLabelingModel
from kashgari.layers import L

from keras_self_attention import SeqSelfAttention


class BLSTMAttentionModel(BaseLabelingModel):
    """Bidirectional LSTM Self Attention Sequence Labeling Model"""
    __architect_name__ = 'BLSTMAttentionModel'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_self_attention': {
                'attention_activation': 'sigmoid'
            },
            'layer_dropout': {
                'rate': 0.5
            },
            'layer_time_distributed': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_self_attention = SeqSelfAttention(**config['layer_self_attention'],
                                                name='layer_self_attention')
        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')

        tensor = layer_blstm(embed_model.output)
        tensor = layer_self_attention(tensor)
        tensor = layer_dropout(tensor)
        output_tensor = layer_time_distributed(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


class SeqSelfAttentionModel(BaseLabelingModel):
    """Bidirectional LSTM Self Attention Sequence Labeling Model"""
    __architect_name__ = 'SeqSelfAttentionModel'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_self_attention_0': {
                'units': 32,
                'attention_activation': 'relu'
            },
            'layer_self_attention_1': {
                'units': 32,
                'attention_activation': 'sigmoid'
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_self_attention_0 = SeqSelfAttention(**config['layer_self_attention_0'],
                                                  name='layer_self_attention0')
        layer_self_attention_1 = SeqSelfAttention(**config['layer_self_attention_1'],
                                                  name='layer_self_attention1')
        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')

        tensor = layer_self_attention_0(embed_model.output)
        tensor = layer_self_attention_1(tensor)
        tensor = layer_dropout(tensor)
        output_tensor = layer_time_distributed(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    print("Hello world")
