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

import logging

from keras.layers import Bidirectional, LSTM
from keras.layers import Dense, Dropout, TimeDistributed, Activation
from keras.models import Model

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


class BLSTMModel(SequenceLabelingModel):
    __architect_name__ = 'BLSTMModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        }, 'dropout_layer': {
            'rate': 0.4
        }
    }

    def _prepare_model(self):
        embed_model = self.embedding.model

        blstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer']))(embed_model.output)
        dropout_layer = Dropout(**self.hyper_parameters['dropout_layer'])(blstm_layer)
        time_distributed_layer = TimeDistributed(Dense(len(self.label2idx)))(dropout_layer)
        activation = Activation('softmax')(time_distributed_layer)

        self.model = Model(embed_model.inputs, activation)

    # TODO: Allow custom loss and optimizer
    def _compile_model(self):
        loss_f = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']

        self.model.compile(loss=loss_f,
                           optimizer=optimizer,
                           metrics=metrics)


if __name__ == '__main__':
    import random
    from keras.callbacks import ModelCheckpoint
    from kashgari.corpus import ChinaPeoplesDailyNerCorpus

    x_train, y_train = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
    x_validate, y_validate = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='validate')
    x_test, y_test = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='test')

    # embedding = WordEmbeddings('sgns.weibo.bigram', sequence_length=100)
    m = BLSTMModel()

    check = ModelCheckpoint('./model.model',
                            monitor='acc',
                            verbose=1,
                            save_best_only=False,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
    m.fit(x_train,
          y_train,
          class_weight=True,
          epochs=1, y_validate=y_validate, x_validate=x_validate, labels_weight=True)

    sample_queries = random.sample(list(range(len(x_train))), 10)
    for i in sample_queries:
        text = x_train[i]
        logging.info('-------- sample {} --------'.format(i))
        logging.info('x: {}'.format(text))
        logging.info('y_true: {}'.format(y_train[i]))
        logging.info('y_pred: {}'.format(m.predict(text)))

    m.evaluate(x_test, y_test)
