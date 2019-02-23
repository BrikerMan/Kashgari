# encoding: utf-8
"""
@author: Alex
@contact: ialexwwang@gmail.com

@version: 0.1
@license: Apache Licence
@file: shadow_models.py
@time: 2019-02-20 16:40
"""
import logging
from keras.layers import Dense, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Model

from kashgari.tasks.classification.base_model import ClassificationModel


class BLSTMModel(ClassificationModel):
    __architect_name__ = 'BLSTMModel'
    __base_hyper_parameters__ = {
            'lstm_layer': {
                'units': 256,
                'return_sequences': False
                },
            'activation_layer': {
                'activation': 'softmax'
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        bilstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer'])
                )(base_model.output)
        dense_layer = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(bilstm_layer)
        output_layers = [dense_layer]

        model = Model(base_model.inputs, output_layers)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class CNNLSTMModel(ClassificationModel):
    __architect_name__ = 'CNNLSTMModel'
    __base_hyper_parameters__ = {
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
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        conv_layer = Conv1D(**self.hyper_parameters['conv_layer'])(base_model.output)
        max_pool_layer = MaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv_layer)
        lstm_layer = LSTM(**self.hyper_parameters['lstm_layer'])(max_pool_layer)
        dense_layer = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(lstm_layer)
        output_layers = [dense_layer]

        model = Model(base_model.inputs, output_layers)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class CNNModel(ClassificationModel):
    __architect_name__ = 'CNNModel'
    __base_hyper_parameters__ = {
            'conv1d_layer': {
                'filters': 128,
                'kernel_size': 5,
                'activation': 'relu'
                },
            'max_pool_layer': {},
            'dense_1_layer': {
                'units': 64,
                'activation': 'relu'
                },
            'activation_layer': {
                'activation': 'softmax'
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        conv1d_layer = Conv1D(**self.hyper_parameters['conv1d_layer'])(base_model.output)
        max_pool_layer = GlobalMaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv1d_layer)
        dense_1_layer = Dense(**self.hyper_parameters['dense_1_layer'])(max_pool_layer)
        dense_2_layer = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(dense_1_layer)

        model = Model(base_model.inputs, dense_2_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()

