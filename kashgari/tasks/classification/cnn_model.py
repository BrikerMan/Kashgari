# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_model.py
@time: 2019-01-21 17:49

"""
import logging
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.models import Model

from kashgari.tasks.classification.base_model import ClassificationModel


class CNNModel(ClassificationModel):
    __architect_name__ = 'CNNModel'
    __base_hyper_parameters__ = {
        'conv1d_layer': {
            'filters': 128,
            'kernel_size': 5,
            'activation': 'relu'
        },
        'max_pool_layer': {},
        'dense_1_layer': {
            'units': 64,
            'activation': 'relu'
        }
    }

    def build_model(self):
        base_model = self.embedding.model
        conv1d_layer = Conv1D(**self.hyper_parameters['conv1d_layer'])(base_model.output)
        max_pool_layer = GlobalMaxPooling1D(**self.hyper_parameters['max_pool_layer'])(conv1d_layer)
        dense_1_layer = Dense(**self.hyper_parameters['dense_1_layer'])(max_pool_layer)
        dense_2_layer = Dense(len(self.label2idx), activation='sigmoid')(dense_1_layer)

        model = Model(base_model.inputs, dense_2_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    from kashgari.corpus import TencentDingdangSLUCorpus

    init_logger()

    x_data, y_data = TencentDingdangSLUCorpus.get_classification_data()
    classifier = CNNModel()
    classifier.fit(x_data, y_data, epochs=1)
    classifier.save('./classifier_saved2')

    model = ClassificationModel.load_model('./classifier_saved2')
    logging.info(model.predict('我要听音乐'))
