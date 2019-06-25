# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-22 11:26

import logging
import tensorflow as tf
from typing import Dict, Any
from kashgari.layers import L
from kashgari.tasks.classification.base_model import BaseClassificationModel


class BiLSTM_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, output_tensor)


class BiGRU_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_gru': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.GRU(**config['layer_bi_lstm']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, output_tensor)


class CNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv1d_layer': {
                'filters': 128,
                'kernel_size': 5,
                'activation': 'relu'
            },
            'max_pool_layer': {},
            'dense_layer': {
                'units': 64,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # build model structure in sequent way
        layers_seq = []
        layers_seq.append(L.Conv1D(**config['conv1d_layer']))
        layers_seq.append(L.GlobalMaxPooling1D(**config['max_pool_layer']))
        layers_seq.append(L.Dense(**config['dense_layer']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class CNN_LSTM_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv_layer': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'max_pool_layer': {
                'pool_size': 2
            },
            'lstm_layer': {
                'units': 100
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_seq = []
        layers_seq.append(L.Conv1D(**config['conv_layer']))
        layers_seq.append(L.MaxPooling1D(**config['max_pool_layer']))
        layers_seq.append(L.LSTM(**config['lstm_layer']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class CNN_GRU_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv_layer': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'max_pool_layer': {
                'pool_size': 2
            },
            'gru_layer': {
                'units': 100
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_seq = []
        layers_seq.append(L.Conv1D(**config['conv_layer']))
        layers_seq.append(L.MaxPooling1D(**config['max_pool_layer']))
        layers_seq.append(L.LSTM(**config['lstm_layer']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


if __name__ == "__main__":
    print(BiLSTM_Model.get_default_hyper_parameters())
    logging.basicConfig(level=logging.DEBUG)
    from kashgari.corpus import SMP2018ECDTCorpus

    x, y = SMP2018ECDTCorpus.load_data()

    m = BiLSTM_Model()
    m.build_model(x, y)
    m.fit(x, y, epochs=5)
    m.evaluate(x, y)
    print(m.predict(x[:10]))
