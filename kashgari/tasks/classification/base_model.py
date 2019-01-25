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
import os
import json
import random
import pathlib
import logging
from typing import Tuple, Dict
import numpy as np

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import to_categorical

from sklearn import metrics
from sklearn.utils import class_weight as class_weight_calculte

import kashgari.macros as k
from kashgari.embeddings import CustomEmbedding, BaseEmbedding
from kashgari.type_hints import *


class ClassificationModel(object):
    base_hyper_parameters = {}

    def __init__(self, embedding: BaseEmbedding = None, hyper_parameters: Dict = None):
        if embedding is None:
            self.embedding = CustomEmbedding('custom', sequence_length=0, embedding_size=100)
        else:
            self.embedding = embedding
        self.model: Model = None
        self.hyper_parameters = self.base_hyper_parameters.copy()
        self._label2idx = {}
        self._idx2label = {}
        if hyper_parameters:
            self.hyper_parameters.update(hyper_parameters)

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
            page_list = list(range(len(x_data) // batch_size + 1))
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
            batch_size: int = 64,
            epochs: int = 5,
            x_validate: List[List[str]] = None,
            y_validate: List[str] = None,
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
        self.build_token2id_label2id_dict(x_train, y_train, x_validate, y_validate)

        if len(x_train) < batch_size:
            batch_size = len(x_train) // 2

        if not self.model:
            if self.embedding.sequence_length == 0:
                self.embedding.sequence_length = sorted([len(y) for y in y_train])[int(0.95*len(y_train))]
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
            fit_kwargs['validation_steps'] = len(x_validate) // batch_size

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

    def predict(self, sentence: Union[List[str], List[List[str]]], batch_size=None):
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
        predict_result = self.model.predict(x, batch_size=batch_size).argmax(-1)
        labels = self.convert_idx_to_label(predict_result)
        if is_list:
            return labels
        else:
            return labels[0]

    def evaluate(self, x_data, y_data, batch_size=None) -> Tuple[float, float, Dict]:
        y_pred = self.predict(x_data, batch_size=batch_size)
        weighted_f1 = metrics.f1_score(y_data, y_pred, average='weighted')
        weighted_recall = metrics.recall_score(y_data, y_pred, average='weighted')
        report = metrics.classification_report(y_data, y_pred, output_dict=True)
        print(metrics.classification_report(y_data, y_pred))
        return weighted_f1, weighted_recall, report

    def save(self, model_path: str):
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)

        with open(os.path.join(model_path, 'labels.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.label2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'words.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.embedding.token2idx, indent=2, ensure_ascii=False))

        self.model.save(os.path.join(model_path, 'model.model'))
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))

    @staticmethod
    def load_model(model_path: str):
        with open(os.path.join(model_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label2idx = json.load(f)

        with open(os.path.join(model_path, 'words.json'), 'r', encoding='utf-8') as f:
            token2idx = json.load(f)

        agent = ClassificationModel()
        agent.model = keras.models.load_model(os.path.join(model_path, 'model.model'))
        agent.embedding.sequence_length = agent.model.input_shape[-1]
        agent.model.summary()
        agent.label2idx = label2idx
        agent.embedding.token2idx = token2idx
        logging.info('loaded model from {}'.format(os.path.abspath(model_path)))
        return agent


if __name__ == "__main__":
    ClassificationModel.load_model('./classifier_saved2')
    print("Hello world")
