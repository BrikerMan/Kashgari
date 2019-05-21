# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 2019-05-20 13:07


from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
from tensorflow import keras

import kashgari
from kashgari import utils
from kashgari.embeddings import BareEmbedding
from kashgari.embeddings.base_embedding import Embedding

L = keras.layers


class BaseLabelingModel(object):
    """Base Sequence Labeling Model"""
    __architect_name__ = 'BLSTMModel'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def __init__(self,
                 embedding: Optional[Embedding] = None,
                 hyper_parameters: Optional[Dict[str, Dict[str, Any]]] = None):
        """

        Args:
            embedding: model embedding
            hyper_parameters: a dict of hyper_parameters.

        Examples:
            You could change customize hyper_parameters like this::

                # get default hyper_parameters
                hyper_parameters = BLSTMModel.get_default_hyper_parameters()
                # change lstm hidden unit to 12
                hyper_parameters['layer_blstm']['units'] = 12
                # init new model with customized hyper_parameters
                labeling_model = BLSTMModel(hyper_parameters=hyper_parameters)
                labeling_model.fit(x, y)
        """
        if embedding is None:
            self.embedding = BareEmbedding(task=kashgari.LABELING)
        else:
            self.embedding = embedding

        self.tf_model: Optional[keras.Model] = None
        self.hyper_parameters = self.get_default_hyper_parameters()
        self._label2idx = {}
        self._idx2label = {}
        self.model_info = {}
        self.pre_processor = self.embedding.processor

        if hyper_parameters:
            self.hyper_parameters.update(hyper_parameters)

    def get_data_generator(self,
                           x_data,
                           y_data,
                           batch_size: int = 64):
        """
        data generator for fit_generator

        Args:
            x_data: Array of feature data (if the model has a single input),
                or tuple of feature data array (if the model has multiple inputs)
            y_data: Array of label data
            batch_size: Number of samples per gradient update, default to 64.

        Returns:
            data generator
        """

        index_list = np.arange(len(x_data[0]))
        page_count = len(x_data) // batch_size + 1

        while True:
            np.random.shuffle(index_list)
            for page in range(page_count):
                start_index = page * batch_size
                end_index = start_index + batch_size
                target_index = index_list[start_index: end_index]

                if len(target_index) == 0:
                    target_index = index_list[0: batch_size]

                x_tensor = self.embedding.process_x_dataset(x_data,
                                                            target_index)
                y_tensor = self.embedding.process_y_dataset(y_data,
                                                            target_index)
                yield (x_tensor, y_tensor)

    def fit(self,
            x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
            y_train: List[List[str]],
            x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
            y_validate: List[List[str]] = None,
            batch_size: int = 64,
            epochs: int = 5,
            fit_kwargs: Dict = None,
            **kwargs):
        """
        Trains the model for a given number of epochs (iterations on a dataset).

        Args:
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Integer. Number of epochs to train the model. default 5.
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit_generator()`` function from
                ``tensorflow.keras.Model`` - https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator
            **kwargs:

        Returns:

        """
        x_train = utils.wrap_as_tuple(x_train)
        y_train = utils.wrap_as_tuple(y_train)
        if x_validate:
            x_validate = utils.wrap_as_tuple(x_validate)
            y_validate = utils.wrap_as_tuple(y_validate)
        if self.embedding.token_count == 0:
            if x_validate is not None:
                x_all = (x_train + x_validate)
                y_all = (y_train + y_validate)
            else:
                x_all = x_train
                y_all = y_train
            self.embedding.analyze_corpus(x_all, y_all)

        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

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

        self.tf_model.fit_generator(train_generator,
                                    steps_per_epoch=len(x_train[0]) // batch_size,
                                    epochs=epochs,
                                    **fit_kwargs)

    def compile_model(self, **kwargs):
        """Configures the model for training.

        Using ``compile()`` function of ``tf.keras.Model`` -
        https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#compile

        Args:
            **kwargs: arguments passed to ``compile()`` function of ``tf.keras.Model``

        Defaults:
            - loss: ``categorical_crossentropy``
            - optimizer: ``adam``
            - metrics: ``['accuracy']``
        """
        if kwargs.get('loss') is None:
            kwargs['loss'] = 'categorical_crossentropy'
        if kwargs.get('optimizer') is None:
            kwargs['optimizer'] = 'adam'
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = ['accuracy']

        self.tf_model.compile(**kwargs)
        self.tf_model.summary()

    def build_model_arc(self):
        raise NotImplementedError


if __name__ == "__main__":
    print("Hello world")
