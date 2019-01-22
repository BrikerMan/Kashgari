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
        model.compile(loss=self.get_weighted_categorical_crossentropy(),
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


if __name__ == '__main__':
    import kashgari as ks
    from keras.callbacks import ModelCheckpoint
    from kashgari.corpus import ChinaPeoplesDailyNerCorpus

    # embedding = ks.embedding.Word2VecEmbedding('sgns.weibo.bigram', limit=1000)
    embedding = ks.embedding.BERTEmbedding('/disk/corpus/bert/chinese_L-12_H-768_A-12/', limit=1000)
    tokenizer = ks.tokenizer.Tokenizer(embedding=embedding,
                                       sequence_length=128)

    x_train, y_train = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
    x_validate, y_validate = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='validate')
    x_test, y_test = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='test')

    m = BLSTMModel(tokenizer=tokenizer)

    check = ModelCheckpoint('./model.model',
                            monitor='acc',
                            verbose=1,
                            save_best_only=False,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
    m.fit(x_train,
          y_train,
          epochs=1,
          x_validate=x_test,
          y_validate=y_test,
          fit_kwargs={'callbacks': [check]})

    m.evaluate(x_test, y_test)
