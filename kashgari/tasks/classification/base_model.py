# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: base_model.py
@time: 2019-01-19 11:50

"""
import logging
import random

import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import to_categorical

from kashgari.data.corpus import Corpus
from kashgari.embedding import CustomEmbedding
from kashgari.tokenizer import Tokenizer
from kashgari.type_hints import *


class ClassificationModel(object):
    def __init__(self):
        self.tokenizer = None
        self.embedding_name = None
        self.model: Model = None

    def prepare_embedding_layer(self):
        embedding = self.tokenizer.embedding
        if isinstance(embedding, CustomEmbedding):
            return Embedding(self.tokenizer.word_num,
                             embedding.embedding_size,
                             input_length=self.tokenizer.sequence_length)
        else:
            return Embedding(len(self.tokenizer.word2idx),
                             self.tokenizer.embedding.embedding_size,
                             input_length=self.tokenizer.sequence_length,
                             weights=[self.tokenizer.embedding.get_embedding_matrix()],
                             trainable=False)

    def build_model(self):
        """
        build model function
        :return:
        """
        raise NotImplementedError()

    def data_generator(self,
                       x_data: Union[List[List[str]], List[str]],
                       y_data: List[str],
                       batch_size: int = 64,
                       is_bert: bool = False):
        while True:
            page_list = list(range(len(x_data) // batch_size + 1))
            random.shuffle(page_list)
            for page in page_list:
                target_x = x_data[page: (page + 1) * batch_size]
                target_y = y_data[page: (page + 1) * batch_size]

                tokenized_x = []
                for x_item in target_x:
                    tokenized_x.append(self.tokenizer.word_to_token(x_item))

                tokenized_y_data = []
                for y_item in target_y:
                    tokenized_y_data.append(self.tokenizer.label_to_token(y_item))

                tokenized_x = sequence.pad_sequences(tokenized_x,
                                                     maxlen=self.tokenizer.sequence_length)
                if is_bert:
                    tokenized_x_seg = np.zeros(shape=(len(tokenized_x), self.tokenizer.sequence_length))
                    tokenized_x_data = [tokenized_x, tokenized_x_seg]
                else:
                    tokenized_x_data = tokenized_x
                tokenized_y_data = to_categorical(tokenized_y_data,
                                                  num_classes=self.tokenizer.class_num,
                                                  dtype=np.int)
                yield (tokenized_x_data, tokenized_y_data)

    def fit(self,
            x_train: ClassificationXType,
            y_train: ClassificationYType,
            tokenizer: Tokenizer = None,
            batch_size: int = 64,
            epochs: int = 5,
            x_validate: ClassificationXType = None,
            y_validate: ClassificationYType = None,
            **kwargs):
        """

        :param x_train: list of training data.
        :param y_train: list of training target label data.
        :param tokenizer: custom tokenizer
        :param batch_size: batch size for trainer model
        :param epochs: Number of epochs to train the model.
        :param x_validate: list of validation data.
        :param y_validate: list of validation target label data.
        :param kwargs:
        :return:
        """
        assert len(x_train) == len(y_train)

        if self.tokenizer is None:
            if tokenizer is None:
                tokenizer = Tokenizer.get_recommend_tokenizer()
            self.tokenizer = tokenizer
        else:
            if tokenizer is not None:
                logging.warning("model already has been set tokenizer, this might cause unexpected result")

        x_data = x_train
        y_data = y_train
        if x_validate:
            x_data += x_validate
            y_data += y_validate
        tokenizer.build_with_corpus(x_data, y_data)

        if len(x_train) < batch_size:
            batch_size = len(x_train) // 2

        if not self.model:
            self.build_model()

        train_generator = self.data_generator(x_train,
                                              y_train,
                                              batch_size,
                                              is_bert=self.tokenizer.is_bert)
        if x_validate:
            validation_generator = self.data_generator(x_validate,
                                                       y_validate,
                                                       batch_size,
                                                       is_bert=self.tokenizer.is_bert)
            kwargs['validation_data'] = validation_generator
            kwargs['validation_steps'] = len(x_validate) // batch_size

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=epochs,
                                 **kwargs)

    def fit_corpus(self,
                   corpus: Corpus,
                   batch_size: int = 128,
                   epochs: int = 10,
                   callbacks=None):
        train_generator = corpus.fit_generator()
        self.tokenizer = corpus.tokenizer
        self.build_model()
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=corpus.data_count // batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks)

    def predict(self, sentence: str):
        tokens = self.tokenizer.word_to_token(sentence)
        padded_tokens = sequence.pad_sequences([tokens], maxlen=self.tokenizer.sequence_length)
        predict_result = self.model.predict(padded_tokens)[0]
        return self.tokenizer.idx2label[predict_result.argmax(0)]


if __name__ == "__main__":
    c = ClassificationModel()
    print("Hello world")
