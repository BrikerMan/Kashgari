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

import keras
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


class CNN_LSTM_Model(object):
    def __init__(self):
        self.embedding: k.Word2VecModels = k.Word2VecModels.sgns_weibo_bigram
        self.tokenizer: Tokenizer = Tokenizer(self.embedding)
        self.tokenizer.build(limit=1000)
        # keras model
        self.model: Model = None

    def _build_embedding_layer_(self) -> Embedding:
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
        model.add(Dense(5,
                        activation='sigmoid'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def preprocess_data(self):
        pass

    def fit(self, x_train, y_train, **kwargs):
        tokenized_x_list = [self.tokenizer.word_to_token(x) for x in x_train]
        inout_x = sequence.pad_sequences(tokenized_x_list,
                                         maxlen=self.tokenizer.sequence_length,
                                         padding='post')
        input_y = keras.utils.to_categorical(y_train,
                                             num_classes=5,
                                             dtype='float32')
        self.model.fit(inout_x, input_y)

    def train(self):
        pass


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    from kashgari.data.data_reader import load_data_from_csv
    init_logger()
    model = CNN_LSTM_Model()
    model.build()
    file_path = '/Users/brikerman/Desktop/ailab/Kashgari/kashgari/data/dataset.csv'
    x, y = load_data_from_csv(file_path)

    model.fit(x, y)
    print("Hello world")
