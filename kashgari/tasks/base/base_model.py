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
import pickle
import pathlib
import traceback
import logging
logger = logging.getLogger(__name__)
import numpy as np
from typing import Dict

import keras
from keras.models import Model
from keras import backend as K
from keras.utils import multi_gpu_model

from kashgari.utils import helper
from kashgari.embeddings import CustomEmbedding, BaseEmbedding
from kashgari.utils.crf import CRF, crf_loss, crf_accuracy
from keras_bert.bert import get_custom_objects as get_bert_custom_objects
from kashgari.layers import AttentionWeightedAverage, KMaxPooling, NonMaskingLayer


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

    def _compile_model(self):
        """
        compile model function
        :return:
        """
        raise NotImplementedError()

    def _prepare_model(self):
        """
        prepare model function
        :return:
        """
        raise NotImplementedError()

    def build_multi_gpu_model(self, gpus: int):
        """
        build multi-gpu model function
        :return:
        """
        if not self.model:
            raise RuntimeError("Model not built yet, Please call build_model function with"
                               "your corpus to build model")

        # If gpus < 2, this will fall back to normal build_model() on CPU or GPU
        if gpus >= 2:
            self.model = multi_gpu_model(self.model, gpus=gpus)
        self._compile_model()
        self.model.summary()

    def save(self, model_path: str):
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)

        model_info = self.info()

        with open(os.path.join(model_path, 'labels.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self._label2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'words.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.embedding.token2idx, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'model.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(model_info, indent=2, ensure_ascii=False))

        with open(os.path.join(model_path, 'struct.json'), 'w', encoding='utf-8') as f:
            f.write(self.model.to_json())

        #self.model.save_weights(os.path.join(model_path, 'weights.h5'))
        optimizer_weight_values = None
        try:
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            optimizer_weight_values = K.batch_get_value(symbolic_weights)
        except Exception as e:
            logger.warn('error occur: {}'.format(e))
            traceback.print_tb(e.__traceback__)
            logger.warn('No optimizer weights found.')
        if optimizer_weight_values is not None:
            with open(os.path.join(model_path, 'optimizer.pkl'), 'wb') as f:
                pickle.dump(optimizer_weight_values, f)

        self.model.save(os.path.join(model_path, 'model.model'))
        logger.info('model saved to {}'.format(os.path.abspath(model_path)))

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
            custom_objects['NonMaskingLayer'] = NonMaskingLayer
            custom_objects.update(get_bert_custom_objects())
        custom_objects['AttentionWeightedAverage'] = AttentionWeightedAverage
        custom_objects['KMaxPooling'] = KMaxPooling
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
        agent.model_info = model_info['model_info']
        if custom_objects:
            logger.debug('prepared custom objects: {}'.format(custom_objects))

        try:
            agent.model = keras.models.load_model(os.path.join(model_path, 'model.model'),
                                                  custom_objects=custom_objects)
        except Exception as e:
            logger.warning('Error `{}` occured trying directly model loading. Try to rebuild.'.format(e))
            logger.debug('Load model structure from json.')
            with open(os.path.join(model_path, 'struct.json'), 'r', encoding='utf-8') as f:
                model_struct = f.read()
            agent.model = keras.models.model_from_json(model_struct,
                                                       custom_objects=custom_objects)
            logger.debug('Build optimizer with model info.')
            optimizer_conf = model_info['hyper_parameters'].get('optimizer', None)
            optimizer = 'adam' #default
            if optimizer_conf is not None and isinstance(optimizer_conf, dict):
                module_str = optimizer_conf.get('module', 'None')
                name_str = optimizer_conf.get('name', 'None')
                params = optimizer_conf.get('params', None)
                invalid_set = [None, 'None', '', {}]
                if not any([module_str.strip() in invalid_set,
                            name_str.strip() in invalid_set,
                            params in invalid_set]):
                    try:
                        optimizer = getattr(eval(module_str), name_str)(**params)
                    except:
                        logger.warn('Invalid optimizer configuration in model info. Use `adam` as default.')
            else:
                logger.warn('No optimizer configuration found in model info. Use `adam` as default.')

            default_compile_params = {'loss': 'categorical_crossentropy', 'metrics':['accuracy']}
            compile_params = model_info['hyper_parameters'].get('compile_params', default_compile_params)
            logger.debug('Compile model from scratch.')
            try:
                agent.model.compile(optimizer=optimizer, **compile_params)
            except:
                logger.warn('Failed to compile model. Compile params seems incorrect.')
                logger.warn('Use default options `{}` to compile.'.format(default_compile_params))
                agent.model.compile(optimizer=optimizer, **default_compile_params)
            logger.debug('Load model weights.')
            agent.model.summary()
            agent.model.load_weights(os.path.join(model_path, 'model.model'))
            agent.model._make_train_function()
            optimizer_weight_values = None
            logger.debug('Load optimizer weights.')
            try:
                with open(os.path.join(model_path, 'optimizer.pkl'), 'rb') as f:
                    optimizer_weight_values = pickle.load(f)
            except Exception as e:
                logger.warn('Try to load optimizer weights but no optimizer weights file found.')
            if optimizer_weight_values is not None:
                agent.model.optimizer.set_weights(optimizer_weight_values)
            else:
                logger.warn('Rebuild model but optimizer weights missed. Retrain needed.')
            logger.info('Model rebuild finished.')
        agent.embedding.update(model_info.get('embedding', {}))
        agent.model.summary()
        agent.label2idx = label2idx
        agent.embedding.token2idx = token2idx
        logger.info('loaded model from {}'.format(os.path.abspath(model_path)))
        return agent


if __name__ == "__main__":
    print("Hello world")