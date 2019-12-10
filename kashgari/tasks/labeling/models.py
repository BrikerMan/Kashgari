# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-20 11:13

import logging
from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.labeling.base_model import BaseLabelingModel
from kashgari.layers import L
from kashgari.layers.crf import CRF

from kashgari.utils import custom_objects

custom_objects['CRF'] = CRF


class BiLSTM_Model(BaseLabelingModel):
    """Bidirectional LSTM Sequence Labeling Model"""

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
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
                                      name='layer_blstm')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


class BiLSTM_CRF_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

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
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
                                      name='layer_blstm')

        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf')

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(BiLSTM_CRF_Model, self).compile_model(**kwargs)


class BiGRU_Model(BaseLabelingModel):
    """Bidirectional GRU Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_bgru': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.GRU(**config['layer_bgru']),
                                      name='layer_bgru')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


class BiGRU_CRF_Model(BaseLabelingModel):
    """Bidirectional GRU CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_bgru': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.GRU(**config['layer_bgru']),
                                      name='layer_bgru')

        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf')

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(BiGRU_CRF_Model, self).compile_model(**kwargs)


class CNN_LSTM_Model(BaseLabelingModel):
    """CNN LSTM Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_conv': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'layer_lstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_conv = L.Conv1D(**config['layer_conv'],
                              name='layer_conv')
        layer_lstm = L.LSTM(**config['layer_lstm'],
                            name='layer_lstm')
        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')
        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        tensor = layer_conv(embed_model.output)
        tensor = layer_lstm(tensor)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from kashgari.corpus import ChineseDailyNerCorpus

    valid_x, valid_y = ChineseDailyNerCorpus.load_data('train')

    model = BiLSTM_CRF_Model()
    model.fit(valid_x, valid_y, epochs=50, batch_size=64)
    model.evaluate(valid_x, valid_y)
