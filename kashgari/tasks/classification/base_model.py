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
from typing import Tuple, Dict

import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.utils import class_weight as class_weight_calculte

from kashgari import macros as k
from kashgari.tasks.base import BaseModel
from kashgari.embeddings import BaseEmbedding
from kashgari.type_hints import *


class ClassificationModel(BaseModel):

    def __init__(self, embedding: BaseEmbedding = None, hyper_parameters: Dict = None, **kwargs):
        super(ClassificationModel, self).__init__(embedding, hyper_parameters, **kwargs)

    @property
    def label2idx(self) -> Dict[str, int]:
        return self._label2idx

    @property
    def token2idx(self) -> Dict[str, int]:
        return self.embedding.token2idx

    @label2idx.setter
    def label2idx(self, value):
        self._label2idx = value
        self._idx2label = dict([(val, key) for (key, val) in value.items()])

    def build_model(self):
        """
        build model function
        :return:
        """
        raise NotImplementedError()

    def build_token2id_label2id_dict(self,
                                     x_train: List[List[str]],
                                     y_train: List[str],
                                     x_validate: List[List[str]] = None,
                                     y_validate: List[str] = None):
        x_data = x_train
        y_data = y_train
        if x_validate:
            x_data += x_validate
            y_data += y_validate
        self.embedding.build_token2idx_dict(x_data, 3)

        label_set = set(y_data)
        label2idx = {
            k.PAD: 0,
        }
        for label in label_set:
            label2idx[label] = len(label2idx)
        self._label2idx = label2idx
        self._idx2label = dict([(val, key) for (key, val) in label2idx.items()])

    def convert_label_to_idx(self, label: Union[List[str], str]) -> Union[List[int], int]:
        if isinstance(label, str):
            return self.label2idx[label]
        else:
            return [self.label2idx[l] for l in label]

    def convert_idx_to_label(self, token: Union[List[int], int]) -> Union[List[str], str]:
        if isinstance(token, int):
            return self._idx2label[token]
        else:
            return [self._idx2label[l] for l in token]

    def get_data_generator(self,
                           x_data: List[List[str]],
                           y_data: List[str],
                           batch_size: int = 64,
                           is_bert: bool = False):
        while True:
            page_list = list(range((len(x_data) // batch_size) + 1))
            random.shuffle(page_list)
            for page in page_list:
                start_index = page * batch_size
                end_index = start_index + batch_size
                target_x = x_data[start_index: end_index]
                target_y = y_data[start_index: end_index]
                if len(target_x) == 0:
                    target_x = x_data[0: batch_size]
                    target_y = y_data[0: batch_size]

                tokenized_x = self.embedding.tokenize(target_x)
                tokenized_y = self.convert_label_to_idx(target_y)

                padded_x = sequence.pad_sequences(tokenized_x,
                                                  maxlen=self.embedding.sequence_length,
                                                  padding='post')
                padded_y = to_categorical(tokenized_y,
                                          num_classes=len(self.label2idx),
                                          dtype=np.int)
                if is_bert:
                    padded_x_seg = np.zeros(shape=(len(padded_x), self.embedding.sequence_length))
                    x_input_data = [padded_x, padded_x_seg]
                else:
                    x_input_data = padded_x
                yield (x_input_data, padded_y)

    def fit(self,
            x_train: List[List[str]],
            y_train: List[str],
            x_validate: List[List[str]] = None,
            y_validate: List[str] = None,
            batch_size: int = 64,
            epochs: int = 5,
            class_weight: bool = False,
            fit_kwargs: Dict = None,
            **kwargs):
        """

        :param x_train: list of training data.
        :param y_train: list of training target label data.
        :param x_validate: list of validation data.
        :param y_validate: list of validation target label data.
        :param batch_size: batch size for trainer model
        :param epochs: Number of epochs to train the model.
        :param class_weight: set class weights for imbalanced classes
        :param fit_kwargs: additional kwargs to be passed to
               :func:`~keras.models.Model.fit`
        :param kwargs:
        :return:
        """
        assert len(x_train) == len(y_train)
        self.build_token2id_label2id_dict(x_train, y_train, x_validate, y_validate)

        if len(x_train) < batch_size:
            batch_size = len(x_train) // 2

        if not self.model:
            if self.embedding.sequence_length == 0:
                self.embedding.sequence_length = sorted([len(x) for x in x_train])[int(0.95 * len(x_train))]
                logging.info('sequence length set to {}'.format(self.embedding.sequence_length))
            self.build_model()

        train_generator = self.get_data_generator(x_train,
                                                  y_train,
                                                  batch_size,
                                                  is_bert=self.embedding.is_bert)

        if fit_kwargs is None:
            fit_kwargs = {}

        if x_validate:
            validation_generator = self.get_data_generator(x_validate,
                                                           y_validate,
                                                           batch_size,
                                                           is_bert=self.embedding.is_bert)
            fit_kwargs['validation_data'] = validation_generator
            fit_kwargs['validation_steps'] = max(len(x_validate) // batch_size, 1)

        if class_weight:
            y_list = self.convert_label_to_idx(y_train)
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

    def _format_output_dic(self, words: List[str], res: np.ndarray):
        results = sorted(list(enumerate(res)), key=lambda x: -x[1])
        candidates = []
        for result in results:
            if float(result[1]) > 0.01:
                candidates.append({
                    'name': self.convert_idx_to_label([result[0]])[0],
                    'confidence': float(result[1]),
                })
        data = {
            'words': words,
            'class': candidates[0],
            'class_candidates': candidates
        }
        return data

    def predict(self,
                sentence: Union[List[str], List[List[str]]],
                batch_size=None,
                output_dict=False,
                debug_info=False) -> Union[List[str], str, List[Dict], Dict]:
        """
        predict with model
        :param sentence: single sentence as List[str] or list of sentence as List[List[str]]
        :param batch_size: predict batch_size
        :param output_dict: return dict with result with confidence
        :param debug_info: print debug info using logging.debug when True
        :return:
        """
        tokens = self.embedding.tokenize(sentence)
        is_list = not isinstance(sentence[0], str)
        if is_list:
            padded_tokens = sequence.pad_sequences(tokens,
                                                   maxlen=self.embedding.sequence_length,
                                                   padding='post')
        else:
            padded_tokens = sequence.pad_sequences([tokens],
                                                   maxlen=self.embedding.sequence_length,
                                                   padding='post')
        if self.embedding.is_bert:
            x = [padded_tokens, np.zeros(shape=(len(padded_tokens), self.embedding.sequence_length))]
        else:
            x = padded_tokens
        res = self.model.predict(x, batch_size=batch_size)
        predict_result = res.argmax(-1)

        if debug_info:
            logging.info('input: {}'.format(x))
            logging.info('output: {}'.format(res))
            logging.info('output argmax: {}'.format(predict_result))

        if output_dict:
            if is_list:
                words_list: List[List[str]] = sentence
            else:
                words_list: List[List[str]] = [sentence]
            results = []
            for index in range(len(words_list)):
                results.append(self._format_output_dic(words_list[index], res[index]))
            if is_list:
                return results
            else:
                return results[0]
        else:
            results = self.convert_idx_to_label(predict_result)
            if is_list:
                return results
            else:
                return results[0]

    def evaluate(self, x_data, y_data, batch_size=None, digits=4, debug_info=False) -> Tuple[float, float, Dict]:
        y_pred = self.predict(x_data, batch_size=batch_size)
        report = metrics.classification_report(y_data, y_pred, output_dict=True, digits=digits)
        print(metrics.classification_report(y_data, y_pred, digits=digits))
        if debug_info:
            for index in random.sample(list(range(len(x_data))), 5):
                logging.debug('------ sample {} ------'.format(index))
                logging.debug('x      : {}'.format(x_data[index]))
                logging.debug('y      : {}'.format(y_data[index]))
                logging.debug('y_pred : {}'.format(y_pred[index]))
        return report
