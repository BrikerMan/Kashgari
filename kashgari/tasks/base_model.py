# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 2019-05-22 11:21


from typing import Dict, Any, List, Optional, Union, Tuple

import os
import json
import pickle
import pathlib
import logging
import numpy as np
from tensorflow import keras
from kashgari import utils
from kashgari.embeddings import BareEmbedding
from kashgari.embeddings.base_embedding import Embedding

L = keras.layers


class BaseModel(object):
    """Base Sequence Labeling Model"""
    __architect_name__ = 'BLSTMModel'
    __task__ = "labeling"

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def info(self):
        return {
            'hyper_parameters': self.hyper_parameters,
            'embedding': self.embedding.info(),
            'architect_name': self.__architect_name__,
            'task': self.__task__,
            'module': self.__module__
        }

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
            self.embedding = BareEmbedding(task=self.__task__)
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

    def build_model(self,
                    x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                    y_train: List[List[str]],
                    x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                    y_validate: List[List[str]] = None,
                    **kwargs):
        self.embedding.analyze_corpus(x_train, y_train)

        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

    def get_data_generator(self,
                           x_data,
                           y_data,
                           batch_size: int = 64,
                           shuffle: bool = True):
        """
        data generator for fit_generator

        Args:
            x_data: Array of feature data (if the model has a single input),
                or tuple of feature data array (if the model has multiple inputs)
            y_data: Array of label data
            batch_size: Number of samples per gradient update, default to 64.
            shuffle:

        Returns:
            data generator
        """
        index_list = np.arange(len(x_data[0]))
        page_count = len(x_data[0]) // batch_size + 1

        while True:
            if shuffle:
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
                ``tensorflow.keras.Model``
                - https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator
            **kwargs:

        Returns:

        """
        self.build_model(x_train, y_train, x_validate, y_validate, **kwargs)

        train_generator = self.get_data_generator(x_train,
                                                  y_train,
                                                  batch_size)
        if fit_kwargs is None:
            fit_kwargs = {}

        if x_validate:
            validation_generator = self.get_data_generator(x_validate,
                                                           y_validate,
                                                           batch_size)

            if isinstance(x_validate, tuple):
                validation_steps = len(x_validate[0]) // batch_size + 1
            else:
                validation_steps = len(x_validate) // batch_size + 1

            fit_kwargs['validation_data'] = validation_generator
            fit_kwargs['validation_steps'] = validation_steps

        if isinstance(x_train, tuple):
            steps_per_epoch = len(x_train[0]) // batch_size + 1
        else:
            steps_per_epoch = len(x_train) // batch_size + 1
        with utils.custom_object_scope():
            self.tf_model.fit_generator(train_generator,
                                        steps_per_epoch=steps_per_epoch,
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

    def predict(self,
                x_data,
                batch_size=None,
                debug_info=False):
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            debug_info: Bool, Should print out the logging info.

        Returns:
            array(s) of predictions.
        """
        with utils.custom_object_scope():
            if isinstance(x_data, tuple):
                lengths = [len(sen) for sen in x_data[0]]
            else:
                lengths = [len(sen) for sen in x_data]
            tensor = self.embedding.process_x_dataset(x_data)
            pred = self.tf_model.predict(tensor, batch_size=batch_size)
            res = self.embedding.reverse_numerize_label_sequences(pred.argmax(-1),
                                                                  lengths)
            if debug_info:
                logging.info('input: {}'.format(tensor))
                logging.info('output: {}'.format(pred))
                logging.info('output argmax: {}'.format(pred.argmax(-1)))
        return res

    def evaluate(self,
                 x_data,
                 y_data,
                 batch_size=None,
                 digits=4,
                 debug_info=False) -> Tuple[float, float, Dict]:
        raise NotImplementedError

    def build_model_arc(self):
        raise NotImplementedError

    def save(self, model_path: str):
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)

        with open(os.path.join(model_path, 'processor.pickle'), 'wb') as f:
            f.write(pickle.dumps(self.embedding.processor))

        with open(os.path.join(model_path, 'model_info.json'), 'w') as f:
            f.write(json.dumps(self.info(), indent=2, ensure_ascii=False))

        self.tf_model.save(os.path.join(model_path, 'model.h5'))
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))


if __name__ == "__main__":
    from kashgari.tasks.labeling import CNNLSTMModel
    from kashgari.corpus import ChineseDailyNerCorpus

    train_x, train_y = ChineseDailyNerCorpus.load_data('valid')

    model = CNNLSTMModel()
    model.build_model(train_x[:100], train_y[:100])
    r = model.predict_entities(train_x[:5])
    model.save('./res')
    import pprint

    pprint.pprint(r)
    model.evaluate(train_x[:20], train_y[:20])
    print("Hello world")

    print(model.predict(train_x[:20]))
