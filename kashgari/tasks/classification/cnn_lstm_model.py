# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_lstm_model.py
@time: 2019-01-19 11:52

"""
from kashgari.utils import k
from kashgari.tokenizer import Tokenizer
import keras.layers as layers

from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


class CNN_LSTM_Model(object):
    def __init__(self):
        self.embedding: k.Word2VecModels = k.Word2VecModels.sgns_weibo_bigram
        self.tokenizer: Tokenizer = Tokenizer(self.embedding)
        self.tokenizer.build()
        self.model: Model = None

    def build_embedding_layer(self) -> Embedding:
        return Embedding(len(self.tokenizer.word2idx),
                         self.tokenizer.embedding_size,
                         input_length=self.tokenizer.sequence_length,
                         weights=[self.tokenizer.get_embedding_matrix()],
                         trainable=False)

    def build(self):
        model = Sequential()
        embedding = self.build_embedding_layer()
        model.add(embedding)
        model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(len(self.tokenizer.label2idx),
                        activation='sigmoid'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self):
        pass


if __name__ == "__main__":
    model = CNN_LSTM_Model()
    print("Hello world")
