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
from keras.layers import Dense, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Model

from kashgari.utils.crf import CRF, crf_loss, crf_accuracy

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


class BLSTMCRFModel(SequenceLabelingModel):
    __architect_name__ = 'BLSTMCRFModel'
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

    def _prepare_model(self):
        base_model = self.embedding.model
        blstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer']))(base_model.output)
        dense_layer = Dense(128, activation='tanh')(blstm_layer)
        crf = CRF(len(self.label2idx), sparse_target=False)
        crf_layer = crf(dense_layer)
        self.model = Model(base_model.inputs, crf_layer)

    # TODO: Allow custom loss and optimizer
    def _compile_model(self):
        self.model.compile(loss=crf_loss,
                           optimizer='adam',
                           metrics=[crf_accuracy])


if __name__ == "__main__":
    print("Hello world")
    from kashgari.utils.logger import init_logger

    init_logger()
    from kashgari.corpus import ChinaPeoplesDailyNerCorpus

    init_logger()

    x_train, y_train = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
    x_validate, y_validate = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='validate')
    x_test, y_test = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='test')

    tagger = BLSTMCRFModel()
    tagger.fit(x_train, y_train, epochs=2)
    tagger.evaluate(x_validate, y_validate)
    tagger.evaluate(x_test, y_test, debug_info=True)

    model = BLSTMCRFModel.load_model('/Users/brikerman/Downloads/KashgariNER.output/model')
    model.evaluate(x_test, y_test, debug_info=True)
