# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_model.py
@time: 2019-01-24 15:26

"""
import logging
from keras.layers import Dense, Conv1D, TimeDistributed, Activation
from keras.models import Model

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


class CNNModel(SequenceLabelingModel):
    __base_hyper_parameters__ = {
        'conv1d_layer': {
            'filters': 128,
            'kernel_size': 5,
            'activation': 'relu'
        }
    }

    def build_model(self):
        base_model = self.embedding.model
        conv1d_layer = Conv1D(**self.hyper_parameters['conv1d_layer'])(base_model.output)
        time_distributed_layer = TimeDistributed(Dense(len(self.label2idx)))(conv1d_layer)
        activation = Activation('softmax')(time_distributed_layer)

        model = Model(base_model.inputs, activation)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


if __name__ == "__main__":
    from kashgari.corpus import ChinaPeoplesDailyNerCorpus

    x_train, y_train = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data()
    x_validate, y_validate = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='validate')
    x_test, y_test = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(data_type='test')

    m = CNNModel()
    m.embedding.sequence_length = 100

    m.fit(x_train,
          y_train,
          epochs=1, y_validate=y_validate, x_validate=x_validate)
    for i in [
        '我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。'
    ]:
        logging.info(list(i))
        logging.info(m.predict(i))

    m.evaluate(x_test, y_test)
