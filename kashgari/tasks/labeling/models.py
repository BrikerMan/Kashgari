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


class BLSTMModel(BaseLabelingModel):
    """Bidirectional LSTM Sequence Labeling Model"""
    __architect_name__ = 'BLSTMModel'

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

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dropout(tensor)
        output_tensor = layer_time_distributed(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


class CNNLSTMModel(BaseLabelingModel):
    """CNN LSTM Sequence Labeling Model"""
    __architect_name__ = 'BLSTMModel'

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

        layer_conv = L.Conv1D(**config['layer_conv'],
                              name='layer_conv')
        layer_lstm = L.LSTM(**config['layer_lstm'],
                            name='layer_lstm')
        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')
        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')

        tensor = layer_conv(embed_model.output)
        tensor = layer_lstm(tensor)
        tensor = layer_dropout(tensor)
        output_tensor = layer_time_distributed(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import os
    import kashgari
    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.embeddings import WordEmbedding

    x, y = ChineseDailyNerCorpus.load_data()

    # # Multi Input model
    # old_fashion_model = CNNLSTMModel()
    # old_fashion_model.fit((x, x), y, epochs=1)
    #
    # # Old fashion model
    # old_fashion_model = CNNLSTMModel()
    # old_fashion_model.fit(x, y, epochs=1)

    # Model For pros 1
    w2v_path = os.path.join(kashgari.utils.get_project_path(), 'tests/test-data/sample_w2v.txt')

    embedding = WordEmbedding(task=kashgari.LABELING,
                              w2v_path=w2v_path,
                              sequence_length='variable')
    labeling1 = BLSTMModel(embedding=embedding)
    labeling1.fit((x, x), y)

    hyper_parameters = BLSTMModel.get_default_hyper_parameters()
    hyper_parameters['layer_blstm']['units'] = 12
    labeling_model = BLSTMModel(hyper_parameters=hyper_parameters)
    labeling_model.fit(x, y)
