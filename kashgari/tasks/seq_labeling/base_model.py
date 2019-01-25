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
import os
import random
import json
import pathlib
import logging
from typing import Tuple, Dict

import numpy as np
import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import to_categorical
from seqeval.metrics import f1_score, classification_report, recall_score

import kashgari.macros as k
from kashgari.utils import helper
from kashgari.embeddings import CustomEmbedding, BaseEmbedding
from kashgari.type_hints import *

from kashgari.utils.crf import CRF, crf_loss


class SequenceLabelingModel(object):
    __base_hyper_parameters__ = {}

    @property
    def hyper_parameters(self):
        return self._hyper_parameters_

    def __init__(self, embedding: BaseEmbedding = None, hyper_parameters: Dict = None):
        if embedding is None:
            self.embedding = CustomEmbedding('custom', sequence_length=0, embedding_size=100)
        else:
            self.embedding = embedding
        self.model: Model = None
        self._hyper_parameters_ = self.__base_hyper_parameters__.copy()
        self._label2idx = {}
        self._idx2label = {}
        if hyper_parameters:
            self._hyper_parameters_.update(hyper_parameters)
        self.model_info = {}

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

    def build_model(self, loss_f=None, optimizer=None, metrics=None, **kwargs):
        """
        build model function
        :return:
        """
        raise NotImplementedError()

    def build_token2id_label2id_dict(self,
                                     x_train: List[List[str]],
                                     y_train: List[List[str]],
                                     x_validate: List[List[str]] = None,
                                     y_validate: List[str] = None):
        x_data = x_train
        y_data = y_train
        if x_validate:
            x_data += x_validate
            y_data += y_validate
        self.embedding.build_token2idx_dict(x_data, 3)

        label_set = []
        for seq in y_data:
            for y in seq:
                if y not in label_set:
                    label_set.append(y)

        label2idx = {
            k.PAD: 0,
            k.BOS: 1,
            k.EOS: 2
        }
        label_set = [i for i in label_set if i not in label2idx]
        for label in label_set:
            label2idx[label] = len(label2idx)

        self.label2idx = label2idx

    def convert_labels_to_idx(self,
                              label: Union[List[List[str]], List[str]],
                              add_eos_bos: bool = True) -> Union[List[List[int]], List[int]]:

        def tokenize_tokens(seq: List[str]):
            tokens = [self._label2idx[i] for i in seq]
            if add_eos_bos:
                tokens = [self._label2idx[k.BOS]] + tokens + [self._label2idx[k.EOS]]
            return tokens

        if isinstance(label[0], str):
            return tokenize_tokens(label)
        else:
            return [tokenize_tokens(l) for l in label]

    def convert_idx_to_labels(self,
                              idx: Union[List[List[int]], List[int]],
                              tokens_length: Union[List[int], int],
                              remove_eos_bos: bool = True) -> Union[List[str], str]:

        def reverse_tokenize_tokens(idx_item, seq_length):
            if remove_eos_bos:
                seq = idx_item[1: 1 + seq_length]
            else:
                seq = idx_item
            tokens = [self._idx2label[i] for i in seq]
            return tokens

        if isinstance(idx[0], int):
            return reverse_tokenize_tokens(idx, tokens_length)
        else:
            labels = []
            for index in range(len(idx)):
                idx_item = idx[index]
                seq_length = tokens_length[index]
                labels.append(reverse_tokenize_tokens(idx_item, seq_length))
            return labels

    def get_data_generator(self,
                           x_data: List[List[str]],
                           y_data: List[List[str]],
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
                tokenized_y = self.convert_labels_to_idx(target_y)

                padded_x = sequence.pad_sequences(tokenized_x,
                                                  maxlen=self.embedding.sequence_length,
                                                  padding='post')
                padded_y = sequence.pad_sequences(tokenized_y,
                                                  maxlen=self.embedding.sequence_length,
                                                  padding='post')

                one_hot_y = to_categorical(padded_y, num_classes=len(self.label2idx))

                if is_bert:
                    padded_x_seg = np.zeros(shape=(len(padded_x), self.embedding.sequence_length))
                    x_input_data = [padded_x, padded_x_seg]
                else:
                    x_input_data = padded_x
                yield (x_input_data, one_hot_y)

    def fit(self,
            x_train: List[List[str]],
            y_train: List[List[str]],
            x_validate: List[List[str]] = None,
            y_validate: List[List[str]] = None,
            batch_size: int = 64,
            epochs: int = 5,
            labels_weight: bool = None,
            default_labels_weight: float = 50.0,
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
        :param labels_weight: set class weights for imbalanced classes
        :param default_labels_weight: default weight for labels not in labels_weight dict
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

            if labels_weight:
                weights = []
                initial_weights = {
                    k.PAD: 1,
                    k.BOS: 1,
                    k.EOS: 1,
                    'O': 1
                }
                for label in self.label2idx.keys():
                    weights.append(initial_weights.get(label, default_labels_weight))
                loss_f = helper.weighted_categorical_crossentropy(np.array(weights))
                self.model_info['loss'] = {
                    'func': 'weighted_categorical_crossentropy',
                    'weights': weights
                }

                self.build_model(loss_f=loss_f, metrics=['categorical_accuracy', 'acc'])
            else:
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

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=epochs,
                                 **fit_kwargs)

    def predict(self, sentence: Union[List[str], List[List[str]]], batch_size=None):
        tokens = self.embedding.tokenize(sentence)
        is_list = not isinstance(sentence[0], str)
        if is_list:
            seq_length = [len(item) for item in sentence]
            padded_tokens = sequence.pad_sequences(tokens,
                                                   maxlen=self.embedding.sequence_length,
                                                   padding='post')
        else:
            seq_length = [len(sentence)]
            padded_tokens = sequence.pad_sequences([tokens],
                                                   maxlen=self.embedding.sequence_length,
                                                   padding='post')
        if self.embedding.is_bert:
            x = [padded_tokens, np.zeros(shape=(len(padded_tokens), self.embedding.sequence_length))]
        else:
            x = padded_tokens
        predict_result = self.model.predict(x, batch_size=batch_size).argmax(-1)
        labels = self.convert_idx_to_labels(predict_result, seq_length)

        if is_list:
            return labels
        else:
            return labels[0]

    def evaluate(self, x_data, y_data, batch_size=None) -> Tuple[float, float, Dict]:
        y_pred = self.predict(x_data, batch_size=batch_size)

        weighted_f1 = f1_score(y_data, y_pred)
        weighted_recall = recall_score(y_data, y_pred)
        report = classification_report(y_data, y_pred)
        print(classification_report(y_data, y_pred))
        return weighted_f1, weighted_recall, report

    def save(self, model_path: str):
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(model_path, 'labels.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.label2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'words.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.embedding.token2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'model.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.model_info, indent=2, ensure_ascii=False))

        self.model.save(os.path.join(model_path, 'model.model'))
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))

    @staticmethod
    def create_custom_objects(model_info):
        custom_objects = {}
        loss = model_info.get('loss')
        if loss and loss['name'] == 'weighted_categorical_crossentropy':
            loss_f = helper.weighted_categorical_crossentropy(np.array(loss['weights']))
            custom_objects['loss'] = loss_f

        if loss and loss['name'] == 'crf':
            custom_objects['CRF'] = CRF
            custom_objects['crf_loss'] = crf_loss
            custom_objects['crf_viterbi_accuracy'] = CRF(128).accuracy

        return custom_objects

    @staticmethod
    def load_model(model_path: str):
        with open(os.path.join(model_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label2idx = json.load(f)

        with open(os.path.join(model_path, 'words.json'), 'r', encoding='utf-8') as f:
            token2idx = json.load(f)

        with open(os.path.join(model_path, 'model.json'), 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        agent = SequenceLabelingModel()
        custom_objects = SequenceLabelingModel.create_custom_objects(model_info)

        if custom_objects:
            logging.debug('prepared custom objects: {}'.format(custom_objects))
        agent.model = keras.models.load_model(os.path.join(model_path, 'model.model'),
                                              custom_objects=custom_objects)
        agent.model.summary()
        agent.embedding.sequence_length = agent.model.input_shape[-1]
        agent.label2idx = label2idx
        agent.embedding.token2idx = token2idx
        logging.info('loaded model from {}'.format(os.path.abspath(model_path)))
        return agent