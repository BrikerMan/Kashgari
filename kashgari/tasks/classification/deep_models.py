# encoding: utf-8
"""
@author: Alex
@contact: ialexwwang@gmail.com

@version: 0.1
@license: Apache Licence
@file: deep_models.py
@time: 2019-02-21 17:54

@Reference: https://github.com/zake7749/DeepToxic/blob/master/sotoxic/models/keras/model_zoo.py
"""
from __future__ import absolute_import, division
import logging

import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Lambda, Activation, Reshape, Flatten
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, GRU, Conv1D
from keras.layers import add, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K

from kashgari.tasks.classification.base_model import ClassificationModel


class AttentionWeightedAverage(Layer):
    '''
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    '''

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)


    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)


    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result


    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)


    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)


    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


class KMaxPooling(Layer):
    '''
    K-max pooling layer that extracts the k-highest activation from a sequence (2nd dimension).
    TensorFlow backend.
    '''

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k


    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))


    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2,1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


class AVCNNModel(ClassificationModel):
    __architect_name__ = 'AVCNNModel'
    __base_hyper_parameters__ = {
            'conv_0': {
                'filters': 300,
                'kernel_size':1,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'conv_1': {
                'filters': 300,
                'kernel_size':2,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'conv_2': {
                'filters': 300,
                'kernel_size':3,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'conv_3': {
                'filters': 300,
                'kernel_size':4,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            # ---
            'attn_0': {},
            'avg_0': {},
            'maxpool_0': {},
            # ---
            'maxpool_1': {},
            'attn_1': {},
            'avg_1': {},
            # ---
            'maxpool_2': {},
            'attn_2': {},
            'avg_2': {},
            # ---
            'maxpool_3': {},
            'attn_3': {},
            'avg_3': {},
            # ---
            'v0_col':{
                #'mode': 'concat',
                'axis': 1
                },
            'v1_col':{
                #'mode': 'concat',
                'axis': 1
                },
            'v2_col':{
                #'mode': 'concat',
                'axis': 1
                },
            'merged_tensor':{
                #'mode': 'concat',
                'axis': 1
                },
            'dropout':{
                'rate': 0.7
                },
            'dense':{
                'units': 144,
                'activation': 'relu'
                },
            'activation_layer':{
                'activation': 'softmax'
                },
            'adam_optimizer':{
                'lr': 1e-3,
                'decay': 1e-7
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        conv_0 = Conv1D(**self.hyper_parameters['conv_0'])(base_model.output)
        conv_1 = Conv1D(**self.hyper_parameters['conv_1'])(base_model.output)
        conv_2 = Conv1D(**self.hyper_parameters['conv_2'])(base_model.output)
        conv_3 = Conv1D(**self.hyper_parameters['conv_3'])(base_model.output)

        maxpool_0 = GlobalMaxPooling1D()(conv_0)
        attn_0 = AttentionWeightedAverage()(conv_0)
        avg_0 = GlobalAveragePooling1D()(conv_0)

        maxpool_1 = GlobalMaxPooling1D()(conv_1)
        attn_1 = AttentionWeightedAverage()(conv_1)
        avg_1 = GlobalAveragePooling1D()(conv_1)

        maxpool_2 = GlobalMaxPooling1D()(conv_2)
        attn_2 = AttentionWeightedAverage()(conv_2)
        avg_2 = GlobalAveragePooling1D()(conv_2)

        maxpool_3 = GlobalMaxPooling1D()(conv_3)
        attn_3 = AttentionWeightedAverage()(conv_3)
        avg_3 = GlobalAveragePooling1D()(conv_3)

        v0_col = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3],
                **self.hyper_parameters['v0_col'])
        v1_col = concatenate([attn_0, attn_1, attn_2, attn_3],
                **self.hyper_parameters['v1_col'])
        v2_col = concatenate([avg_1, avg_2, avg_0, avg_3],
                **self.hyper_parameters['v2_col'])
        merged_tensor = concatenate([v0_col, v1_col, v2_col],
                **self.hyper_parameters['merged_tensor'])
        output = Dropout(**self.hyper_parameters['dropout'])(merged_tensor)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(output)

        model = Model(base_model.inputs, output)
        adam_optimizer = optimizers.Adam(**self.hyper_parameters['adam_optimizer'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class KMaxCNNModel(ClassificationModel):
    __architect_name__ = 'KMaxCNNModel'
    __base_hyper_parameters__ = {
            'conv_0': {
                'filters': 180,
                'kernel_size': 1,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'conv_1': {
                'filters': 180,
                'kernel_size': 2,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'conv_2': {
                'filters': 180,
                'kernel_size': 3,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'conv_3': {
                'filters': 180,
                'kernel_size': 4,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
                },
            'maxpool_0': {
                'k': 3
                },
            'maxpool_1': {
                'k': 3
                },
            'maxpool_2': {
                'k': 3
                },
            'maxpool_3': {
                'k': 3
                },
            'merged_tensor': {
                #'mode': 'concat',
                'axis': 1
                },
            'dropout':{
                'rate': 0.6
                },
            'dense':{
                'units': 144,
                'activation': 'relu'
                },
            'activation_layer':{
                'activation': 'softmax'
                },
            'adam_optimizer':{
                'lr': 1e-3,
                'decay': 1e-7
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        conv_0 = Conv1D(**self.hyper_parameters['conv_0'])(base_model.output)
        conv_1 = Conv1D(**self.hyper_parameters['conv_1'])(base_model.output)
        conv_2 = Conv1D(**self.hyper_parameters['conv_2'])(base_model.output)
        conv_3 = Conv1D(**self.hyper_parameters['conv_3'])(base_model.output)

        maxpool_0 = KMaxPooling(**self.hyper_parameters['maxpool_0'])(conv_0)
        maxpool_1 = KMaxPooling(**self.hyper_parameters['maxpool_1'])(conv_1)
        maxpool_2 = KMaxPooling(**self.hyper_parameters['maxpool_2'])(conv_2)
        maxpool_3 = KMaxPooling(**self.hyper_parameters['maxpool_3'])(conv_3)

        merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3],
                **self.hyper_parameters['merged_tensor'])
        output = Dropout(**self.hyper_parameters['dropout'])(merged_tensor)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(output)

        model = Model(base_model.inputs, output)
        adam_optimizer = optimizers.Adam(**self.hyper_parameters['adam_optimizer'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class RCNNModel(ClassificationModel):
    __architect_name__ = 'RCNNModel'
    __base_hyper_parameters__ = {
            'rnn_0': {
                'units': 64,
                'return_sequences': True
                },
            'conv_0': {
                'filters': 128,
                'kernel_size': 2,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
                },
            'maxpool': {},
            'attn': {},
            'average': {},
            'concat': {
                'axis': 1
                },
            'dropout':{
                'rate': 0.5
                },
            'dense':{
                'units': 120,
                'activation': 'relu'
                },
            'activation_layer':{
                'activation': 'softmax'
                },
            'adam_optimizer':{
                'lr': 1e-3,
                'clipvalue': 5,
                'decay': 1e-5
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        rnn_0 = Bidirectional(GRU(**self.hyper_parameters['rnn_0']))(base_model.output)
        conv_0 = Conv1D(**self.hyper_parameters['conv_0'])(rnn_0)
        maxpool = GlobalMaxPooling1D()(conv_0)
        attn = AttentionWeightedAverage()(conv_0)
        average = GlobalAveragePooling1D()(conv_0)

        concatenated = concatenated([maxpool, attn, average],
                **self.hyper_parameters['concat'])
        output = Dropout(**self.hyper_parameters['dropout'])(concatenated)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(output)

        model = Model(base_model.inputs, output)
        adam_optimizer = optimizers.Adam(**self.hyper_parameters['adam_optimizer'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class AVRNNModel(ClassificationModel):
    __architect_name__ = 'AVRNNModel'
    __base_hyper_parameters__ = {
            'rnn_0': {
                'units': 60,
                'return_sequences': True
                },
            'rnn_1': {
                'units': 60,
                'return_sequences': True
                },
            'concat_rnn': {
                'axis': 2
                },
            'last': {},
            'maxpool': {},
            'attn': {},
            'average': {},
            'all_views':{
                'axis': 1
                },
            'dropout':{
                'rate': 0.5
                },
            'dense':{
                'units': 144,
                'activation': 'relu'
                },
            'activation_layer':{
                'activation': 'softmax'
                },
            'adam_optimizer':{
                'lr': 1e-3,
                'clipvalue': 5,
                'decay': 1e-6
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        rnn_0 = Bidirectional(GRU(**self.hyper_parameters['rnn_0']))(base_model.output)
        rnn_1 = Bidirectional(GRU(**self.hyper_parameters['rnn_1']))(rnn_0)
        concat_rnn = concatenate([rnn_0, rnn_1],
                **self.hyper_parameters['concat_rnn'])

        last = Lambda(lambda t: t[:, -1], name='last')(concat_rnn)
        maxpool = GlobalMaxPooling1D()(concat_rnn)
        attn = AttentionWeightedAverage()(concat_rnn)
        average = GlobalAveragePooling1D()(concat_rnn)

        all_views = concatenate([last, maxpool, attn, average],
                **self.hyper_parameters['all_views'])
        output = Dropout(**self.hyper_parameters['dropout'])(all_views)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(output)

        model = Model(base_model.inputs, output)
        adam_optimizer = optimizers.Adam(**self.hyper_parameters['adam_optimizer'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class DropoutBGRUModel(ClassificationModel):
    __architect_name__ = 'DropoutBGRUModel'
    __base_hyper_parameters__ = {
            'rnn_0': {
                'units': 64,
                'return_sequences': True
                },
            'dropout_rnn':{
                'rate': 0.35
                },
            'rnn_1': {
                'units': 64,
                'return_sequences': True
                },
            'last': {},
            'maxpool': {},
            'average': {},
            'all_views': {
                'axis': 1
                },
            'dropout':{
                'rate': 0.5
                },
            'dense':{
                'units': 72,
                'activation': 'relu'
                },
            'activation_layer':{
                'activation': 'softmax'
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        rnn_0 = Bidirectional(GRU(**self.hyper_parameters['rnn_0']))(base_model.output)
        dropout_rnn = Dropout(**self.hyper_parameters['dropout_rnn'])(rnn_0)
        rnn_1 = Bidirectional(GRU(**self.hyper_parameters['rnn_1']))(dropout_rnn)
        last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)
        maxpool = GlobalMaxPooling1D()(rnn_1)
        #attn = AttentionWeightedAverage()(rnn_1)
        average = GlobalAveragePooling1D()(rnn_1)

        all_views = concatenate([last, maxpool, average],
                **self.hyper_parameters['all_views'])
        output = Dropout(**self.hyper_parameters['dropout'])(all_views)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(output)

        model = Model(base_model.inputs, output)
        #adam_optimizer = optimizers.Adam(**self.hyper_parameters['adam_optimizer'])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


class DropoutAVRNNModel(ClassificationModel):
    __architect_name__ = 'DropoutAVRNNModel'
    __base_hyper_parameters__ = {
            'rnn_0': {
                'units': 56,
                'return_sequences': True
                },
            'rnn_dropout': {
                'rate': 0.3
                },
            'rnn_1': {
                'units': 56,
                'return_sequences': True
                },
            'last': {},
            'maxpool': {},
            'attn': {},
            'average': {},
            'all_views':{
                'axis': 1
                },
            'dropout_0':{
                'rate': 0.5
                },
            'dense':{
                'units': 128,
                'activation': 'relu'
                },
            'dropout_1':{
                'rate': 0.25
                },
            'activation_layer':{
                'activation': 'softmax'
                },
            'adam_optimizer':{
                'lr': 1e-3,
                'clipvalue': 5,
                'decay': 1e-7
                }
            }

    def build_model(self):
        base_model = self.embedding.model
        rnn_0 = Bidirectional(GRU(**self.hyper_parameters['rnn_0']))(base_model.output)
        rnn_dropout = SpatialDropout1D(**self.hyper_parameters['rnn_dropout'])(rnn_0)
        rnn_1 = Bidirectional(GRU(**self.hyper_parameters['rnn_1']))(rnn_dropout)

        last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)
        maxpool = GlobalMaxPooling1D()(rnn_1)
        attn = AttentionWeightedAverage()(rnn_1)
        average = GlobalAveragePooling1D()(rnn_1)

        all_views = concatenate([last, maxpool, attn, average],
                **self.hyper_parameters['all_views'])
        output = Dropout(**self.hyper_parameters['dropout_0'])(all_views)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dropout(**self.hyper_parameters['dropout_1'])(output)
        output = Dense(len(self.label2idx),
                **self.hyper_parameters['activation_layer'])(output)

        model = Model(base_model.inputs, output)
        adam_optimizer = optimizers.Adam(**self.hyper_parameters['adam_optimizer'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()

