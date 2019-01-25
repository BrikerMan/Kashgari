# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: blstm_crf_model.py
@time: 2019-01-23 17:02

"""
from keras.layers import Dense, Bidirectional, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Model

from kashgari.utils.crf import CRF, crf_loss

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


class BLSTMCRFModel(SequenceLabelingModel):
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dense_layer': {
            'units': 64,
            'activation': 'tanh'
        }
    }

    def build_model(self):
        base_model = self.embedding.model
        blstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer']))(base_model.output)
        dense_layer = Dense(128, activation='tanh')(blstm_layer)
        crf = CRF(len(self.label2idx), sparse_target=True)
        crf_layer = crf(dense_layer)

        model = Model(base_model.inputs, crf_layer)
        model.compile(loss=crf_loss,
                      optimizer='adam',
                      metrics=[crf.accuracy])
        self.model = model
        self.model.summary()
        self.model_info['loss'] = {
            'name': 'crf'
        }


if __name__ == "__main__":
    print("Hello world")
    from kashgari.utils.logger import init_logger
    from kashgari.corpus import ChinaPeoplesDailyNerCorpus

    init_logger()

    x_train, y_train = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
    x_validate, y_validate = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='validate')
    x_test, y_test = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='test')

    classifier = BLSTMCRFModel()
    classifier.fit(x_train, y_train, epochs=2)
    classifier.evaluate(x_validate, y_validate)
    classifier.evaluate(x_test, y_train)
