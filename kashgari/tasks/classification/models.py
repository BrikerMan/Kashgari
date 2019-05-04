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

import keras
from keras.layers import Bidirectional, Conv1D
from keras.layers import Dense, Lambda, Flatten
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import concatenate
from keras.models import Model

from kashgari.layers import AttentionWeightedAverage, KMaxPooling, LSTMLayer, GRULayer
from kashgari.tasks.classification.base_model import ClassificationModel


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
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 0.0
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        conv1d_layer = Conv1D(**self.hyper_parameters['conv1d_layer'])(base_model.output)
        max_pool_layer = GlobalMaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv1d_layer)
        dense_1_layer = Dense(**self.hyper_parameters['dense_1_layer'])(max_pool_layer)
        dense_2_layer = Dense(len(self.label2idx), **self.hyper_parameters['activation_layer'])(dense_1_layer)

        self.model = Model(base_model.inputs, dense_2_layer)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class BLSTMModel(ClassificationModel):
    __architect_name__ = 'BLSTMModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': False
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 0.0
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        blstm_layer = Bidirectional(LSTMLayer(**self.hyper_parameters['lstm_layer']))(base_model.output)
        dense_layer = Dense(len(self.label2idx), **self.hyper_parameters['activation_layer'])(blstm_layer)
        output_layers = [dense_layer]

        self.model = Model(base_model.inputs, output_layers)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


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
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 0.0
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        conv_layer = Conv1D(**self.hyper_parameters['conv_layer'])(base_model.output)
        max_pool_layer = MaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv_layer)
        lstm_layer = LSTMLayer(**self.hyper_parameters['lstm_layer'])(max_pool_layer)
        dense_layer = Dense(len(self.label2idx),
                            **self.hyper_parameters['activation_layer'])(lstm_layer)
        output_layers = [dense_layer]

        self.model = Model(base_model.inputs, output_layers)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class AVCNNModel(ClassificationModel):
    __architect_name__ = 'AVCNNModel'
    __base_hyper_parameters__ = {
        'spatial_dropout': {
            'rate': 0.25
        },
        'conv_0': {
            'filters': 300,
            'kernel_size': 1,
            'kernel_initializer': 'normal',
            'padding': 'valid',
            'activation': 'relu'
        },
        'conv_1': {
            'filters': 300,
            'kernel_size': 2,
            'kernel_initializer': 'normal',
            'padding': 'valid',
            'activation': 'relu'
        },
        'conv_2': {
            'filters': 300,
            'kernel_size': 3,
            'kernel_initializer': 'normal',
            'padding': 'valid',
            'activation': 'relu'
        },
        'conv_3': {
            'filters': 300,
            'kernel_size': 4,
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
        'v0_col': {
            # 'mode': 'concat',
            'axis': 1
        },
        'v1_col': {
            # 'mode': 'concat',
            'axis': 1
        },
        'v2_col': {
            # 'mode': 'concat',
            'axis': 1
        },
        'merged_tensor': {
            # 'mode': 'concat',
            'axis': 1
        },
        'dropout': {
            'rate': 0.7
        },
        'dense': {
            'units': 144,
            'activation': 'relu'
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 1e-7
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        embedded_seq = SpatialDropout1D(**self.hyper_parameters['spatial_dropout'])(base_model.output)
        conv_0 = Conv1D(**self.hyper_parameters['conv_0'])(embedded_seq)
        conv_1 = Conv1D(**self.hyper_parameters['conv_1'])(embedded_seq)
        conv_2 = Conv1D(**self.hyper_parameters['conv_2'])(embedded_seq)
        conv_3 = Conv1D(**self.hyper_parameters['conv_3'])(embedded_seq)

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

        self.model = Model(base_model.inputs, output)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class KMaxCNNModel(ClassificationModel):
    __architect_name__ = 'KMaxCNNModel'
    __base_hyper_parameters__ = {
        'spatial_dropout': {
            'rate': 0.2
        },
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
            # 'mode': 'concat',
            'axis': 1
        },
        'dropout': {
            'rate': 0.6
        },
        'dense': {
            'units': 144,
            'activation': 'relu'
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 1e-7
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        embedded_seq = SpatialDropout1D(**self.hyper_parameters['spatial_dropout'])(base_model.output)
        conv_0 = Conv1D(**self.hyper_parameters['conv_0'])(embedded_seq)
        conv_1 = Conv1D(**self.hyper_parameters['conv_1'])(embedded_seq)
        conv_2 = Conv1D(**self.hyper_parameters['conv_2'])(embedded_seq)
        conv_3 = Conv1D(**self.hyper_parameters['conv_3'])(embedded_seq)

        maxpool_0 = KMaxPooling(**self.hyper_parameters['maxpool_0'])(conv_0)
        # maxpool_0f = Reshape((-1,))(maxpool_0)
        maxpool_0f = Flatten()(maxpool_0)
        maxpool_1 = KMaxPooling(**self.hyper_parameters['maxpool_1'])(conv_1)
        # maxpool_1f = Reshape((-1,))(maxpool_1)
        maxpool_1f = Flatten()(maxpool_1)
        maxpool_2 = KMaxPooling(**self.hyper_parameters['maxpool_2'])(conv_2)
        # maxpool_2f = Reshape((-1,))(maxpool_2)
        maxpool_2f = Flatten()(maxpool_2)
        maxpool_3 = KMaxPooling(**self.hyper_parameters['maxpool_3'])(conv_3)
        # maxpool_3f = Reshape((-1,))(maxpool_3)
        maxpool_3f = Flatten()(maxpool_3)
        # maxpool_0 = GlobalMaxPooling1D()(conv_0)
        # maxpool_1 = GlobalMaxPooling1D()(conv_1)
        # maxpool_2 = GlobalMaxPooling1D()(conv_2)
        # maxpool_3 = GlobalMaxPooling1D()(conv_3)

        # merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3],
        #                            **self.hyper_parameters['merged_tensor'])
        merged_tensor = concatenate([maxpool_0f, maxpool_1f, maxpool_2f, maxpool_3f],
                                    **self.hyper_parameters['merged_tensor'])
        # flatten = Reshape((-1,))(merged_tensor)
        # output = Dropout(**self.hyper_parameters['dropout'])(flatten)
        output = Dropout(**self.hyper_parameters['dropout'])(merged_tensor)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                       **self.hyper_parameters['activation_layer'])(output)

        self.model = Model(base_model.inputs, output)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class RCNNModel(ClassificationModel):
    __architect_name__ = 'RCNNModel'
    __base_hyper_parameters__ = {
        'spatial_dropout': {
            'rate': 0.2
        },
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
        'dropout': {
            'rate': 0.5
        },
        'dense': {
            'units': 120,
            'activation': 'relu'
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'clipvalue': 5,
                'decay': 1e-5
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        embedded_seq = SpatialDropout1D(**self.hyper_parameters['spatial_dropout'])(base_model.output)
        rnn_0 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_0']))(embedded_seq)
        conv_0 = Conv1D(**self.hyper_parameters['conv_0'])(rnn_0)
        maxpool = GlobalMaxPooling1D()(conv_0)
        attn = AttentionWeightedAverage()(conv_0)
        average = GlobalAveragePooling1D()(conv_0)

        concatenated = concatenate([maxpool, attn, average],
                                   **self.hyper_parameters['concat'])
        output = Dropout(**self.hyper_parameters['dropout'])(concatenated)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                       **self.hyper_parameters['activation_layer'])(output)

        self.model = Model(base_model.inputs, output)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class AVRNNModel(ClassificationModel):
    __architect_name__ = 'AVRNNModel'
    __base_hyper_parameters__ = {
        'spatial_dropout': {
            'rate': 0.25
        },
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
        'all_views': {
            'axis': 1
        },
        'dropout': {
            'rate': 0.5
        },
        'dense': {
            'units': 144,
            'activation': 'relu'
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'clipvalue': 5,
                'decay': 1e-6
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        embedded_seq = SpatialDropout1D(**self.hyper_parameters['spatial_dropout'])(base_model.output)
        rnn_0 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_0']))(embedded_seq)
        rnn_1 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_1']))(rnn_0)
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

        self.model = Model(base_model.inputs, output)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class DropoutBGRUModel(ClassificationModel):
    __architect_name__ = 'DropoutBGRUModel'
    __base_hyper_parameters__ = {
        'spatial_dropout': {
            'rate': 0.15
        },
        'rnn_0': {
            'units': 64,
            'return_sequences': True
        },
        'dropout_rnn': {
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
        'dropout': {
            'rate': 0.5
        },
        'dense': {
            'units': 72,
            'activation': 'relu'
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 0.0
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        embedded_seq = SpatialDropout1D(**self.hyper_parameters['spatial_dropout'])(base_model.output)
        rnn_0 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_0']))(embedded_seq)
        dropout_rnn = Dropout(**self.hyper_parameters['dropout_rnn'])(rnn_0)
        rnn_1 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_1']))(dropout_rnn)
        last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)
        maxpool = GlobalMaxPooling1D()(rnn_1)
        # attn = AttentionWeightedAverage()(rnn_1)
        average = GlobalAveragePooling1D()(rnn_1)

        all_views = concatenate([last, maxpool, average],
                                **self.hyper_parameters['all_views'])
        output = Dropout(**self.hyper_parameters['dropout'])(all_views)
        output = Dense(**self.hyper_parameters['dense'])(output)
        output = Dense(len(self.label2idx),
                       **self.hyper_parameters['activation_layer'])(output)

        self.model = Model(base_model.inputs, output)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


class DropoutAVRNNModel(ClassificationModel):
    __architect_name__ = 'DropoutAVRNNModel'
    __base_hyper_parameters__ = {
        'spatial_dropout': {
            'rate': 0.25
        },
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
        'all_views': {
            'axis': 1
        },
        'dropout_0': {
            'rate': 0.5
        },
        'dense': {
            'units': 128,
            'activation': 'relu'
        },
        'dropout_1': {
            'rate': 0.25
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'clipvalue': 5,
                'decay': 1e-7
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        embedded_seq = SpatialDropout1D(**self.hyper_parameters['spatial_dropout'])(base_model.output)
        rnn_0 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_0']))(embedded_seq)
        rnn_dropout = SpatialDropout1D(**self.hyper_parameters['rnn_dropout'])(rnn_0)
        rnn_1 = Bidirectional(GRULayer(**self.hyper_parameters['rnn_1']))(rnn_dropout)

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

        self.model = Model(base_model.inputs, output)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])


if __name__ == '__main__':
    from kashgari.corpus import TencentDingdangSLUCorpus
    from kashgari.embeddings import WordEmbeddings, BERTEmbedding

    train_x, train_y = TencentDingdangSLUCorpus.get_classification_data()

    w2v = WordEmbeddings('sgns.weibo.bigram',
                         sequence_length=15,
                         limit=5000)
    bert = BERTEmbedding('bert-base-chinese', sequence_length=15)
    t_model = CNNModel(bert)
    t_model.fit(train_x, train_y, epochs=1)
