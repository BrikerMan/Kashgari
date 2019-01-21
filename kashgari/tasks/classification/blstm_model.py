# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: blstm_model.py
@time: 2019-01-21 17:37

"""
from keras.layers import Dense, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Model

from kashgari.tasks.classification.base_model import ClassificationModel


class BLSTMModel(ClassificationModel):
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': False
        }
    }

    def build_model(self):
        current, input_layers = self.prepare_embedding_layer()

        blstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer']))(current)
        dense_layer = Dense(len(self.tokenizer.label2idx), activation='sigmoid')(blstm_layer)
        output_layers = [dense_layer]

        model = Model(input_layers, output_layers)
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

    classifier = BLSTMModel()
    classifier.fit(x_data, y_data, batch_size=2, class_weight=True)
