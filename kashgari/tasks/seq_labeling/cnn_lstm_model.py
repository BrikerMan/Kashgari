# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_lstm_model.py
@time: 2019-01-24 15:27

"""
from keras.layers import Dense, Conv1D, TimeDistributed, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


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
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        conv_layer = Conv1D(**self.hyper_parameters['conv_layer'])(base_model.output)
        # max_pool_layer = MaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv_layer)
        lstm_layer = LSTM(**self.hyper_parameters['lstm_layer'])(conv_layer)
        time_distributed_layer = TimeDistributed(Dense(len(self.label2idx)))(lstm_layer)
        activation = Activation('softmax')(time_distributed_layer)
        output_layers = [activation]

        self.model = Model(base_model.inputs, output_layers)

    # TODO: Allow custom loss and optimizer
    def _compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])


if __name__ == "__main__":
    print("Hello world")
    from kashgari.utils.logger import init_logger
    from kashgari.corpus import ChinaPeoplesDailyNerCorpus

    init_logger()

    x_train, y_train = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
    x_validate, y_validate = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='validate')
    x_test, y_test = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='test')

    classifier = CNNLSTMModel()
    classifier.fit(x_train, y_train, epochs=2)
    classifier.evaluate(x_validate, y_validate)
    classifier.evaluate(x_test, y_train)
