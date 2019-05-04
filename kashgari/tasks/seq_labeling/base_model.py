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
from keras.preprocessing import sequence
from keras.utils import to_categorical, multi_gpu_model
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities

import kashgari.macros as k
from kashgari.utils import helper
from kashgari.type_hints import *

from kashgari.tasks.base import BaseModel
from kashgari.embeddings import BaseEmbedding


class SequenceLabelingModel(BaseModel):

    def __init__(self, embedding: BaseEmbedding = None, hyper_parameters: Dict = None, **kwargs):
        super(SequenceLabelingModel, self).__init__(embedding, hyper_parameters, **kwargs)
        self.task = 'sequence_labeling'

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

    def _prepare_model(self):
        """
        prepare model function
        :return:
        """
        raise NotImplementedError()

    def _compile_model(self):
        """
        compile model function
        :return:
        """
        raise NotImplementedError()

    def build_model(self,
                    x_train: List[List[str]],
                    y_train: List[List[str]],
                    x_validate: List[List[str]] = None,
                    y_validate: List[List[str]] = None,
                    labels_weight: bool = None,
                    default_labels_weight: float = 50.0,
                    ):
        assert len(x_train) == len(y_train)
        self.build_token2id_label2id_dict(x_train, y_train, x_validate, y_validate)

        if not self.model:
            if self.embedding.sequence_length == 0:
                self.embedding.sequence_length = sorted([len(x) for x in x_train])[int(0.95 * len(x_train))]
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

        self._prepare_model()
        self._compile_model()
        self.model.summary()

    def build_token2id_label2id_dict(self,
                                     x_train: List[List[str]],
                                     y_train: List[List[str]],
                                     x_validate: List[List[str]] = None,
                                     y_validate: List[List[str]] = None):
        for index in range(len(x_train)):
            assert len(x_train[index]) == len(y_train[index])
        x_data = x_train
        y_data = y_train
        if x_validate:
            x_data = x_train + x_validate
            y_data = y_data + y_validate
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
                if k.config.sequence_labeling_tokenize_add_bos_eos:
                    tokens = [self._label2idx[k.BOS]] + tokens + [self._label2idx[k.EOS]]
                else:
                    tokens = [self._label2idx[k.NO_TAG]] + tokens + [self._label2idx[k.NO_TAG]]
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
                           batch_size: int = 64):
        is_bert = self.embedding.embedding_type == 'bert'
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
                tokenized_y = self.convert_labels_to_idx(target_y)

                padded_x = sequence.pad_sequences(tokenized_x,
                                                  maxlen=self.embedding.sequence_length,
                                                  padding='post', truncating='post')
                padded_y = sequence.pad_sequences(tokenized_y,
                                                  maxlen=self.embedding.sequence_length,
                                                  padding='post', truncating='post')

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
        if not self.model:
            self.build_model(x_train, y_train, x_validate, y_validate, labels_weight, default_labels_weight)
        if len(x_train) < batch_size:
            batch_size = len(x_train) // 2

        train_generator = self.get_data_generator(x_train,
                                                  y_train,
                                                  batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}

        if x_validate:
            validation_generator = self.get_data_generator(x_validate,
                                                           y_validate,
                                                           batch_size)

            fit_kwargs['validation_data'] = validation_generator
            fit_kwargs['validation_steps'] = len(x_validate) // batch_size

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=epochs,
                                 **fit_kwargs)

    def _format_output_dic(self, words: List[str], tags: List[str], chunk_joiner: str):
        chunks = get_entities(tags)
        res = {
            'words': words,
            'entities': []
        }
        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            entity = {
                'text': chunk_joiner.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                # 'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)
        return res

    def predict(self,
                sentence: Union[List[str], List[List[str]]],
                batch_size=None,
                output_dict=False,
                chunk_joiner=' ',
                debug_info=False):
        """
        predict with model
        :param sentence: input for predict, accept a single sentence as type List[str] or
                         list of sentence as List[List[str]]
        :param batch_size: predict batch_size
        :param output_dict: return dict with result with confidence
        :param chunk_joiner: the char to join the chunks when output dict
        :param debug_info: print debug info using logging.debug when True
        :return:
        """
        tokens = self.embedding.tokenize(sentence)
        is_list = not isinstance(sentence[0], str)
        if is_list:
            seq_length = [len(item) for item in sentence]
            padded_tokens = sequence.pad_sequences(tokens,
                                                   maxlen=self.embedding.sequence_length,
                                                   padding='post', truncating='post')
        else:
            seq_length = [len(sentence)]
            padded_tokens = sequence.pad_sequences([tokens],
                                                   maxlen=self.embedding.sequence_length,
                                                   padding='post', truncating='post')
        if self.embedding.is_bert:
            x = [padded_tokens, np.zeros(shape=(len(padded_tokens), self.embedding.sequence_length))]
        else:
            x = padded_tokens

        predict_result_prob = self.model.predict(x, batch_size=batch_size)
        predict_result = predict_result_prob.argmax(-1)
        if debug_info:
            logging.info('input: {}'.format(x))
            logging.info('output: {}'.format(predict_result_prob))
            logging.info('output argmax: {}'.format(predict_result))

        result: List[List[str]] = self.convert_idx_to_labels(predict_result, seq_length)
        if output_dict:
            dict_list = []
            if is_list:
                sentence_list: List[List[str]] = sentence
            else:
                sentence_list: List[List[str]] = [sentence]
            for index in range(len(sentence_list)):
                dict_list.append(self._format_output_dic(sentence_list[index],
                                                         result[index],
                                                         chunk_joiner))
            if is_list:
                return dict_list
            else:
                return dict_list[0]
        else:
            if is_list:
                return result
            else:
                return result[0]

    def evaluate(self, x_data, y_data, batch_size=None, digits=4, debug_info=False) -> Tuple[float, float, Dict]:
        seq_length = [len(x) for x in x_data]
        tokenized_y = self.convert_labels_to_idx(y_data)
        padded_y = sequence.pad_sequences(tokenized_y,
                                          maxlen=self.embedding.sequence_length,
                                          padding='post', truncating='post')
        y_true = self.convert_idx_to_labels(padded_y, seq_length)
        y_pred = self.predict(x_data, batch_size=batch_size)
        if debug_info:
            for index in random.sample(list(range(len(x_data))), 5):
                logging.debug('------ sample {} ------'.format(index))
                logging.debug('x      : {}'.format(x_data[index]))
                logging.debug('y_true : {}'.format(y_true[index]))
                logging.debug('y_pred : {}'.format(y_pred[index]))
        report = classification_report(y_true, y_pred, digits=digits)
        print(classification_report(y_true, y_pred, digits=digits))
        return report
