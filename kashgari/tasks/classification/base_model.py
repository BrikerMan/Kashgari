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
import random
import numpy as np
from typing import Tuple, Dict
from keras.layers import Embedding, Input, Layer
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import to_categorical

from kashgari.utils import helper
from kashgari.tokenizer import Tokenizer
from kashgari.type_hints import *
from kashgari.embedding import CustomEmbedding, BERTEmbedding

from sklearn.metrics import classification_report
from sklearn.utils import class_weight as class_weight_calculte


class ClassificationModel(object):
    __base_hyper_parameters__ = {}

    @property
    def hyper_parameters(self):
        return self._hyper_parameters_

    def __init__(self, tokenizer: Tokenizer = None, hyper_parameters: Dict = None):
        self.tokenizer = tokenizer
        self.model: Model = None
        self._hyper_parameters_ = self.__base_hyper_parameters__.copy()
        if hyper_parameters:
            self._hyper_parameters_.update(hyper_parameters)

    def prepare_embedding_layer(self) -> Tuple[Layer, List[Layer]]:
        embedding = self.tokenizer.embedding
        if isinstance(embedding, CustomEmbedding):
            input_x = Input(shape=(self.tokenizer.sequence_length,), dtype='int32')
            current = Embedding(self.tokenizer.word_num,
                                embedding.embedding_size)(input_x)
            input_layers = [input_x]

        elif isinstance(embedding, BERTEmbedding):
            base_model = embedding.get_base_model(self.tokenizer.sequence_length)
            input_layers = base_model.inputs
            current = base_model.output
            current = helper.NonMaskingLayer()(current)

        else:
            input_x = Input(shape=(self.tokenizer.sequence_length,), dtype='int32')
            current = Embedding(len(self.tokenizer.word2idx),
                                self.tokenizer.embedding.embedding_size,
                                input_length=self.tokenizer.sequence_length,
                                weights=[self.tokenizer.embedding.get_embedding_matrix()],
                                trainable=False)(input_x)

            input_layers = [input_x]
        return current, input_layers

    def build_model(self):
        """
        build model function
        :return:
        """
        raise NotImplementedError()

    def prepare_tokenizer_if_needs(self,
                                   x_train: ClassificationXType,
                                   y_train: ClassificationYType,
                                   x_validate: ClassificationXType = None,
                                   y_validate: ClassificationYType = None,):
        if self.tokenizer is None:
            tokenizer = Tokenizer.get_recommend_tokenizer()
            self.tokenizer = tokenizer

        x_data = x_train
        y_data = y_train
        if x_validate:
            x_data += x_validate
            y_data += y_validate
        self.tokenizer.build_with_corpus(x_data, y_data)

    def get_data_generator(self,
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
            batch_size: int = 64,
            epochs: int = 5,
            x_validate: ClassificationXType = None,
            y_validate: ClassificationYType = None,
            class_weight: bool = False,
            fit_kwargs: Dict = None,
            **kwargs):
        """

        :param x_train: list of training data.
        :param y_train: list of training target label data.
        :param batch_size: batch size for trainer model
        :param epochs: Number of epochs to train the model.
        :param x_validate: list of validation data.
        :param y_validate: list of validation target label data.
        :param y_validate: list of validation target label data.
        :param y_validate: list of validation target label data.
        :param class_weight: set class weights for imbalanced classes
        :param fit_kwargs: additional kwargs to be passed to
               :func:`~keras.models.Model.fit`
        :return:
        """
        assert len(x_train) == len(y_train)
        self.prepare_tokenizer_if_needs(x_train, y_train, x_validate, y_validate)

        if len(x_train) < batch_size:
            batch_size = len(x_train) // 2

        if not self.model:
            self.build_model()

        train_generator = self.get_data_generator(x_train,
                                                  y_train,
                                                  batch_size,
                                                  is_bert=self.tokenizer.is_bert)
        if x_validate:
            validation_generator = self.get_data_generator(x_validate,
                                                           y_validate,
                                                           batch_size,
                                                           is_bert=self.tokenizer.is_bert)
            kwargs['validation_data'] = validation_generator
            kwargs['validation_steps'] = len(x_validate) // batch_size

        if fit_kwargs is None:
            fit_kwargs = {}

        if class_weight:
            y_list = [self.tokenizer.label2idx[y] for y in y_train]
            class_weights = class_weight_calculte.compute_class_weight('balanced',
                                                                       np.unique(y_list),
                                                                       y_list)
        else:
            class_weights = None

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=epochs,
                                 class_weight=class_weights,
                                 **fit_kwargs)

    def predict(self, sentence: str):
        tokens = self.tokenizer.word_to_token(sentence)
        padded_tokens = sequence.pad_sequences([tokens], maxlen=self.tokenizer.sequence_length)
        if self.tokenizer.is_bert:
            x = [padded_tokens, np.zeros(shape=(1, self.tokenizer.sequence_length))]
        else:
            x = padded_tokens
        predict_result = self.model.predict(x)[0]
        return self.tokenizer.idx2label[predict_result.argmax(0)]

    def evaluate(self, x_data, y_data, batch_size: int = 128):
        y_true = np.array([self.tokenizer.label2idx[y] for y in y_data])

        tokenized_x = []
        for x_item in x_data:
            tokenized_x.append(self.tokenizer.word_to_token(x_item))

        tokenized_x = sequence.pad_sequences(tokenized_x,
                                             maxlen=self.tokenizer.sequence_length)
        if self.tokenizer.is_bert:
            tokenized_x_seg = np.zeros(shape=(len(tokenized_x), self.tokenizer.sequence_length))
            tokenized_x_data = [tokenized_x, tokenized_x_seg]
        else:
            tokenized_x_data = tokenized_x

        y_pred = self.model.predict(tokenized_x_data, batch_size=128)

        y_pred = y_pred.argmax(1)

        target_names = list(self.tokenizer.idx2label.values())
        print((classification_report(y_true,
                                     y_pred,
                                     target_names=target_names,
                                     labels=range(len(self.tokenizer.label2idx)))))


if __name__ == "__main__":
    c = ClassificationModel()
    print("Hello world")
