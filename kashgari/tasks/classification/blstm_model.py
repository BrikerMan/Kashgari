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
        base_model = self.embedding.model
        blstm_layer = Bidirectional(LSTM(**self.hyper_parameters['lstm_layer']))(base_model.output)
        dense_layer = Dense(len(self.label2idx), activation='sigmoid')(blstm_layer)
        output_layers = [dense_layer]

        model = Model(base_model.inputs, output_layers)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    from kashgari.embeddings import BERTEmbedding
    from kashgari.corpus import TencentDingdangSLUCorpus
    import jieba

    init_logger()

    x_data, y_data = TencentDingdangSLUCorpus.get_classification_data(max_count=50)
    x_data = [list(jieba.cut(x)) for x in x_data]
    embedding = BERTEmbedding('bert-base-chinese', sequence_length=10)
    classifier = BLSTMModel(embedding)
    classifier.fit(x_data, y_data, epochs=1)
    sentence = list('语言学包含了几种分支领域。')
    print(classifier.predict(sentence))

    import logging
    from kashgari.embeddings import WordEmbeddings
    embedding = WordEmbeddings('sgns.weibo.bigram', sequence_length=30, limit=5000)
    model = BLSTMModel(embedding=embedding)
    model.fit(x_data, y_data)
    sentence = list('语言学包含了几种分支领域。')
    logging.info(model.embedding.tokenize(sentence))
    logging.info(model.predict(sentence))
    self.assertTrue(isinstance(self.model.predict(sentence), str))
    self.assertTrue(isinstance(self.model.predict([sentence]), list))
