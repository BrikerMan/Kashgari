# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: blstm_model
@time: 2019-01-21

"""
from keras.layers import Bidirectional, LSTM
from keras.layers import Dense, Dropout, TimeDistributed, Activation
from keras.models import Model

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


class BLSTMModel(SequenceLabelingModel):
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        }, 'dropout_layer': {
            'rate': 0.4
        }
    }

    def build_model(self):
        """
        build model function
        :return:
        """
        current, input_layers = self.prepare_embedding_layer()

        blstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer']))(current)
        dropout_layer = Dropout(**self.hyper_parameters['dropout_layer'])(blstm_layer)
        time_distributed_layer = TimeDistributed(Dense(self.tokenizer.class_num))(dropout_layer)
        activation = Activation('softmax')(time_distributed_layer)

        model = Model(input_layers, activation)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


if __name__ == '__main__':
    print("hello, world")
