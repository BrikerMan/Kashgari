# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: base_model.py
@time: 2019-01-27 14:53

"""
import os
import json
import pathlib
import logging
import numpy as np
from typing import Dict

import keras
from keras.models import Model
from kashgari.utils import helper
from kashgari.embeddings import CustomEmbedding, BaseEmbedding
from kashgari.utils.crf import CRF, crf_loss, crf_accuracy
from keras_bert.bert import get_custom_objects as get_bert_custom_objects


class BaseModel(object):
    __base_hyper_parameters__ = {}
    __architect_name__ = ''

    @property
    def hyper_parameters(self):
        return self._hyper_parameters_

    def __init__(self, embedding: BaseEmbedding = None, hyper_parameters: Dict = None, **kwargs):
        if embedding is None:
            self.embedding = CustomEmbedding('custom', sequence_length=0, embedding_size=100)
        else:
            self.embedding = embedding
        self.model: Model = None
        self._hyper_parameters_ = self.__base_hyper_parameters__.copy()
        self._label2idx = {}
        self._idx2label = {}
        self.model_info = {}

        self.task = 'classification'

        if hyper_parameters:
            self._hyper_parameters_.update(hyper_parameters)

    def info(self):
        return {
            'architect_name': self.__architect_name__,
            'task': self.task,
            'embedding': self.embedding.info(),
            'hyper_parameters': self.hyper_parameters,
            'model_info': self.model_info
        }

    def save(self, model_path: str):
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)

        model_info = self.info()

        with open(os.path.join(model_path, 'labels.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self._label2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'words.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.embedding.token2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'model.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(model_info, indent=2, ensure_ascii=False))

        self.model.save(os.path.join(model_path, 'model.model'))
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))

    @staticmethod
    def create_custom_objects(model_info):
        custom_objects = {}
        loss = model_info.get('model_info', {}).get('loss')
        if loss and loss['name'] == 'weighted_categorical_crossentropy':
            loss_f = helper.weighted_categorical_crossentropy(np.array(loss['weights']))
            custom_objects['loss'] = loss_f

        architect_name = model_info.get('architect_name')
        if architect_name and 'CRF' in architect_name:
            custom_objects['CRF'] = CRF
            custom_objects['crf_loss'] = crf_loss
            custom_objects['crf_accuracy'] = crf_accuracy

        embedding = model_info.get('embedding')

        if embedding and embedding['embedding_type'] == 'bert':
            custom_objects['NonMaskingLayer'] = helper.NonMaskingLayer
            custom_objects.update(get_bert_custom_objects())

        return custom_objects

    @classmethod
    def load_model(cls, model_path: str):
        with open(os.path.join(model_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label2idx = json.load(f)

        with open(os.path.join(model_path, 'words.json'), 'r', encoding='utf-8') as f:
            token2idx = json.load(f)

        with open(os.path.join(model_path, 'model.json'), 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        agent = cls()
        custom_objects = cls.create_custom_objects(model_info)

        if custom_objects:
            logging.debug('prepared custom objects: {}'.format(custom_objects))

        agent.model = keras.models.load_model(os.path.join(model_path, 'model.model'),
                                              custom_objects=custom_objects)
        agent.embedding.update(model_info.get('embedding', {}))
        agent.model.summary()
        agent.label2idx = label2idx
        agent.embedding.token2idx = token2idx
        logging.info('loaded model from {}'.format(os.path.abspath(model_path)))
        return agent


if __name__ == "__main__":
    print("Hello world")
