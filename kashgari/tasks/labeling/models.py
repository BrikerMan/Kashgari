# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-20 11:13

from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.labeling.base_model import BaseLabelingModel

L = keras.layers


class BLSTMModel(BaseLabelingModel):
    """Bidirectional LSTM Sequence Labeling Model"""
    __architect_name__ = 'BLSTMModel'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
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

    def prepare_model_arc(self):

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

    def prepare_model_arc(self):
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
    import logging

    logging.basicConfig(level=logging.DEBUG)

    from kashgari.corpus import ChineseDailyNerCorpus

    x, y = ChineseDailyNerCorpus.load_data()

    # Old fashion model
    old_fashion_model = CNNLSTMModel()
    old_fashion_model.fit(x, y)

    # # Model For pros 1
    # embedding = BareEmbedding(sequence_length=20)
    # embedding.prepare_for_labeling(x, y)
    # embedding.processor.save_dicts('./cached_processor')
    #
    # print(embedding.embed_model.summary())
    # labeling1 = BLSTMModel(embedding=embedding)
    # labeling1.prepare_model_arc()
    # labeling1.compile_model()
    #
    # # Model For pros 1
    # processor = PreProcessor.load_cached_processor('./cached_processor')
    # embedding = BareEmbedding(sequence_length='variable', processor=processor)
    #
    # labeling2 = BLSTMModel(embedding=embedding)
    # labeling2.prepare_model_arc()
    # labeling2.compile_model()
    #
    # hyper_parameters = BLSTMModel.get_default_hyper_parameters()
    # hyper_parameters['layer_blstm']['units'] = 12
    # labeling_model = BLSTMModel(hyper_parameters=hyper_parameters)
    # labeling_model.fit(x, y)
    #
