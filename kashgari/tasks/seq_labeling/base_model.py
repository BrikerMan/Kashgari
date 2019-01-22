# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: base_model
@time: 2019-01-21

"""
import random
import logging
from typing import Tuple, Dict

import numpy as np
from keras.layers import Embedding, Input, Layer
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import to_categorical

from seqeval.metrics import classification_report

from kashgari import k
from kashgari.embedding import CustomEmbedding, BERTEmbedding
from kashgari.tokenizer import Tokenizer
from kashgari.type_hints import *
from kashgari.utils import helper


class SequenceLabelingModel(object):
    __base_hyper_parameters__ = {}

    __task_type__ = k.TaskType.tagging

    @property
    def hyper_parameters(self):
        return self._hyper_parameters_

    def __init__(self, tokenizer: Tokenizer = None, hyper_parameters: Dict = None):
        self.tokenizer = tokenizer
        self.model: Model = None
        self.tokenizer.task_type = self.__task_type__
        self._hyper_parameters_ = self.__base_hyper_parameters__.copy()
        if hyper_parameters:
            self._hyper_parameters_.update(hyper_parameters)

    def build_model(self):
        """
        build model function
        :return:
        """
        raise NotImplementedError()

    def get_weighted_categorical_crossentropy(self):
        weights = [5] * len(self.tokenizer.label2idx)
        weights[0] = 1
        weights = np.array(weights)
        loss_f = helper.weighted_categorical_crossentropy(weights)
        return loss_f

    def prepare_embedding_layer(self) -> Tuple[Layer, List[Layer]]:
        embedding = self.tokenizer.embedding
        if isinstance(embedding, CustomEmbedding):
            # input_x = Input(shape=(self.tokenizer.sequence_length, ), dtype='int32')
            # current = Embedding(self.tokenizer.word_num,
            #                     50,
            #                     input_length=self.tokenizer.sequence_length)(input_x)
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

    def prepare_tokenizer_if_needs(self,
                                   x_train: ClassificationXType,
                                   y_train: ClassificationYType,
                                   x_validate: ClassificationXType = None,
                                   y_validate: ClassificationYType = None):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.get_recommend_tokenizer()

        x_data = x_train
        y_data = y_train
        if x_validate:
            x_data += x_validate
            y_data += y_validate

        self.tokenizer.build_with_corpus(x_data, y_data, task=k.TaskType.tagging)

        for i in range(5):
            logging.info('sample x {} : {} -> {}'.format(i, x_train[i], self.tokenizer.word_to_token(x_train[i])))
            logging.info('sample y {} : {} -> {}'.format(i, y_train[i], self.tokenizer.label_to_token(y_train[i])))

    def get_data_generator(self,
                           x_data: Union[List[List[str]], List[str]],
                           y_data: List[str],
                           batch_size: int = 64,
                           is_bert: bool = False):
        while True:
            page_list = list(range(len(x_data) // batch_size + 1))
            random.shuffle(page_list)
            for page in page_list:
                start_index = page * batch_size
                end_index = start_index + batch_size

                target_x = x_data[start_index: end_index]
                target_y = y_data[start_index: end_index]

                tokenized_x = []
                for x_item in target_x:
                    tokenized_x.append(self.tokenizer.word_to_token(x_item))

                tokenized_y_data = []
                for y_item in target_y:
                    if isinstance(y_item, str):
                        y_item = y_item.split(' ')
                    tokenized_y_data.append(self.tokenizer.label_to_token(y_item))

                tokenized_x = sequence.pad_sequences(tokenized_x,
                                                     maxlen=self.tokenizer.sequence_length,
                                                     padding='post')
                if is_bert:
                    tokenized_x_seg = np.zeros(shape=(len(tokenized_x), self.tokenizer.sequence_length))
                    tokenized_x_data = [tokenized_x, tokenized_x_seg]
                else:
                    tokenized_x_data = tokenized_x
                tokenized_y_data = sequence.pad_sequences(tokenized_y_data,
                                                          maxlen=self.tokenizer.sequence_length,
                                                          padding='post')
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
        :param fit_kwargs: additional kwargs to be passed to
        :func:`~keras.models.Model.fit`
        :return:
        """
        if fit_kwargs is None:
            fit_kwargs = {}

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
            fit_kwargs['validation_data'] = validation_generator
            fit_kwargs['validation_steps'] = len(x_validate) // batch_size

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=epochs,
                                 **fit_kwargs)

    def predict(self, sentence: str):
        tokens = self.tokenizer.word_to_token(sentence)
        padded_tokens = sequence.pad_sequences([tokens],
                                               maxlen=self.tokenizer.sequence_length,
                                               padding='post')
        if self.tokenizer.is_bert:
            x = [padded_tokens, np.zeros(shape=(1, self.tokenizer.sequence_length))]
        else:
            x = padded_tokens
        predict_result = self.model.predict(x)
        predict_tokens = predict_result.argmax(-1)[0][:len(tokens)-2]
        return self.tokenizer.token_to_label(list(predict_tokens))

    def evaluate(self, x_data, y_data, batch_size: int = 128):
        tokenized_x = []
        sequence_length = []
        for x_item in x_data:
            tokens = self.tokenizer.word_to_token(x_item)
            tokenized_x.append(tokens)
            sequence_length.append(len(tokens) - 2)

        tokenized_x_padding = sequence.pad_sequences(tokenized_x,
                                                     maxlen=self.tokenizer.sequence_length,
                                                     padding='post')
        if self.tokenizer.is_bert:
            tokenized_x_seg = np.zeros(shape=(len(tokenized_x_padding), self.tokenizer.sequence_length))
            tokenized_x_data = [tokenized_x_padding, tokenized_x_seg]
        else:
            tokenized_x_data = tokenized_x_padding

        y_pred_array = self.model.predict(tokenized_x_data, batch_size=128).argmax(-1)
        y_true = []
        for index, item in enumerate(y_pred_array):
            y_true.append(self.tokenizer.token_to_label(list(item)[: len(tokenized_x[index])]))

        logging.info('\n{}'.format(classification_report(y_true,
                                                         y_data)))

        if __name__ == '__main__':
            pass
