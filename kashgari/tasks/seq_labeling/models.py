# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: models.py
@time: 2019-02-28 17:19

"""
import keras
from keras.layers import Dense, Conv1D, TimeDistributed, Activation, Bidirectional, Dropout
from keras.models import Model

from kashgari.layers import LSTMLayer
from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel
from kashgari.utils.crf import CRF, crf_loss as kashgari_crf_loss, crf_accuracy as kashgari_crf_accuracy


class CNNLSTMModel(SequenceLabelingModel):
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
            'units': 100,
            'return_sequences': True
        },
        'time_distributed_layer': {

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

    def build_model(self):
        base_model = self.embedding.model
        conv_layer = Conv1D(**self.hyper_parameters['conv_layer'])(base_model.output)
        lstm_layer = LSTMLayer(**self.hyper_parameters['lstm_layer'])(conv_layer)
        time_distributed_layer = TimeDistributed(Dense(len(self.label2idx)),
                                                 **self.hyper_parameters['time_distributed_layer'])(lstm_layer)
        activation_layer = Activation(**self.hyper_parameters['activation_layer'])(time_distributed_layer)

        model = Model(base_model.inputs, [activation_layer])
        optimizer_class = getattr(eval(self.hyper_parameters['optimizer']['module']),
                                  self.hyper_parameters['optimizer']['name'])
        optimizer = optimizer_class(**self.hyper_parameters['optimizer']['params'])
        model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])
        self.model = model
        self.model.summary()


class BLSTMModel(SequenceLabelingModel):
    __architect_name__ = 'BLSTMModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dropout_layer': {
            'rate': 0.4
        },
        'time_distributed_layer': {

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

    def build_model(self):
        """
        build model function
        :return:
        """
        embed_model = self.embedding.model

        blstm_layer = Bidirectional(LSTMLayer(**self.hyper_parameters['lstm_layer']))(embed_model.output)
        dropout_layer = Dropout(**self.hyper_parameters['dropout_layer'])(blstm_layer)
        time_distributed_layer = TimeDistributed(Dense(len(self.label2idx)),
                                                 **self.hyper_parameters['time_distributed_layer'])(dropout_layer)
        activation_layer = Activation(**self.hyper_parameters['activation_layer'])(time_distributed_layer)

        model = Model(embed_model.inputs, activation_layer)
        optimizer_class = getattr(eval(self.hyper_parameters['optimizer']['module']),
                                  self.hyper_parameters['optimizer']['name'])
        optimizer = optimizer_class(**self.hyper_parameters['optimizer']['params'])
        model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])
        self.model = model
        self.model.summary()


class BLSTMCRFModel(SequenceLabelingModel):
    __architect_name__ = 'BLSTMCRFModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dense_layer': {
            'units': 128,
            'activation': 'tanh'
        },
        'crf_layer': {
            'sparse_target': False
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
            'loss': 'kashgari_crf_loss',
            # 'optimizer': 'adam',
            'metrics': ['kashgari_crf_accuracy']
        }
    }

    def build_model(self):
        loss = self.hyper_parameters['compile_params']['loss']
        if loss.startswith('kashgari'):
            loss = eval(loss)

        metrics = self.hyper_parameters['compile_params']['metrics']
        new_metrics = []
        for met in metrics:
            if met.startswith('kashgari') :
                new_metrics.append(eval(met))
            else:
                new_metrics.append(met)
        metrics = new_metrics

        base_model = self.embedding.model
        blstm_layer = Bidirectional(LSTMLayer(**self.hyper_parameters['lstm_layer']))(base_model.output)
        dense_layer = Dense(**self.hyper_parameters['dense_layer'])(blstm_layer)
        crf = CRF(len(self.label2idx), **self.hyper_parameters['crf_layer'])
        crf_layer = crf(dense_layer)
        model = Model(base_model.inputs, crf_layer)
        optimizer_class = getattr(eval(self.hyper_parameters['optimizer']['module']),
                                  self.hyper_parameters['optimizer']['name'])
        optimizer = optimizer_class(**self.hyper_parameters['optimizer']['params'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        self.model.summary()


if __name__ == "__main__":
    print("Hello world")
