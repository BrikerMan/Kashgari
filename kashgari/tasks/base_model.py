# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 2019-05-22 11:21

import os
import json
import logging
import warnings
import pathlib
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import kashgari

from kashgari import utils
from kashgari.embeddings import BareEmbedding
from kashgari.embeddings.base_embedding import Embedding

L = keras.layers


class BaseModel(object):
    """Base Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def info(self):
        model_json_str = self.tf_model.to_json()

        return {
            'config': {
                'hyper_parameters': self.hyper_parameters,
            },
            'tf_model': json.loads(model_json_str),
            'embedding': self.embedding.info(),
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'tf_version': tf.__version__,
            'kashgari_version': kashgari.__version__
        }

    @property
    def task(self):
        return self.embedding.task

    @property
    def token2idx(self) -> Dict[str, int]:
        return self.embedding.token2idx

    @property
    def label2idx(self) -> Dict[str, int]:
        return self.embedding.label2idx

    @property
    def pre_processor(self):
        warnings.warn("The 'pre_processor' property is deprecated, "
                      "use 'processor' instead", DeprecationWarning, 2)
        """Deprecated. Use `self.processor` instead."""
        return self.embedding.processor

    @property
    def processor(self):
        return self.embedding.processor

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

        self.tf_model: keras.Model = None
        self.hyper_parameters = self.get_default_hyper_parameters()
        self.model_info = {}

        if hyper_parameters:
            self.hyper_parameters.update(hyper_parameters)

    def build_model(self,
                    x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                    y_train: Union[List[List[str]], List[str]],
                    x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                    y_validate: Union[List[List[str]], List[str]] = None):
        """
        Build model with corpus

        Args:
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data

        Returns:

        """

        if x_validate is not None and not isinstance(x_validate, tuple):
            self.embedding.analyze_corpus(x_train + x_validate, y_train + y_validate)
        else:
            self.embedding.analyze_corpus(x_train, y_train)

        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

    def build_multi_gpu_model(self,
                              gpus: int,
                              x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                              y_train: Union[List[List[str]], List[str]],
                              cpu_merge: bool = True,
                              cpu_relocation: bool = False,
                              x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                              y_validate: Union[List[List[str]], List[str]] = None):
        """
        Build multi-GPU model with corpus

        Args:
            gpus: Integer >= 2, number of on GPUs on which to create model replicas.
            cpu_merge: A boolean value to identify whether to force merging model weights
                under the scope of the CPU or not.
            cpu_relocation: A boolean value to identify whether to create the model's weights
                under the scope of the CPU. If the model is not defined under any preceding device
                scope, you can still rescue it by activating this option.
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data

        Returns:

        """
        if x_validate is not None and not isinstance(x_validate, tuple):
            self.embedding.analyze_corpus(x_train + x_validate, y_train + y_validate)
        else:
            self.embedding.analyze_corpus(x_train, y_train)

        if self.tf_model is None:
            with utils.custom_object_scope():
                self.build_model_arc()
                self.tf_model = tf.keras.utils.multi_gpu_model(self.tf_model,
                                                               gpus,
                                                               cpu_merge=cpu_merge,
                                                               cpu_relocation=cpu_relocation)
                self.compile_model()

    def build_tpu_model(self, strategy: tf.contrib.distribute.TPUStrategy,
                        x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                        y_train: Union[List[List[str]], List[str]],
                        x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                        y_validate: Union[List[List[str]], List[str]] = None):
        """
        Build TPU model with corpus

        Args:
            strategy: `TPUDistributionStrategy`. The strategy to use for replicating model
                across multiple TPU cores.
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data

        Returns:

        """

        if x_validate is not None and not isinstance(x_validate, tuple):
            self.embedding.analyze_corpus(x_train + x_validate, y_train + y_validate)
        else:
            self.embedding.analyze_corpus(x_train, y_train)
        if self.tf_model is None:
            with utils.custom_object_scope():
                self.build_model_arc()
                self.tf_model = tf.contrib.tpu.keras_to_tpu_model(self.tf_model, strategy=strategy)
                self.compile_model(optimizer=tf.train.AdamOptimizer())

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
        index_list = np.arange(len(x_data))
        page_count = len(x_data) // batch_size + 1

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
            y_train: Union[List[List[str]], List[str]],
            x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
            y_validate: Union[List[List[str]], List[str]] = None,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List[keras.callbacks.Callback] = None,
            fit_kwargs: Dict = None,
            shuffle: bool = True):
        """
        Trains the model for a given number of epochs with fit_generator (iterations on a dataset).

        Args:
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Integer. Number of epochs to train the model. default 5.
            callbacks:
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit_generator()`` function from
                ``tensorflow.keras.Model``
                - https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator
            shuffle:

        Returns:

        """
        self.build_model(x_train, y_train, x_validate, y_validate)
        train_generator = self.get_data_generator(x_train,
                                                  y_train,
                                                  batch_size,
                                                  shuffle)
        if fit_kwargs is None:
            fit_kwargs = {}

        validation_generator = None
        validation_steps = None
        if x_validate:
            validation_generator = self.get_data_generator(x_validate,
                                                           y_validate,
                                                           batch_size,
                                                           shuffle)

            if isinstance(x_validate, tuple):
                validation_steps = len(x_validate[0]) // batch_size + 1
            else:
                validation_steps = len(x_validate) // batch_size + 1

        if isinstance(x_train, tuple):
            steps_per_epoch = len(x_train[0]) // batch_size + 1
        else:
            steps_per_epoch = len(x_train) // batch_size + 1
        with utils.custom_object_scope():
            return self.tf_model.fit_generator(train_generator,
                                               steps_per_epoch=steps_per_epoch,
                                               epochs=epochs,
                                               validation_data=validation_generator,
                                               validation_steps=validation_steps,
                                               callbacks=callbacks,
                                               **fit_kwargs)

    def fit_without_generator(self,
                              x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                              y_train: Union[List[List[str]], List[str]],
                              x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                              y_validate: Union[List[List[str]], List[str]] = None,
                              batch_size: int = 64,
                              epochs: int = 5,
                              callbacks: List[keras.callbacks.Callback] = None,
                              fit_kwargs: Dict = None):
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
            callbacks:
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit_generator()`` function from
                ``tensorflow.keras.Model``
                - https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator

        Returns:

        """
        self.build_model(x_train, y_train, x_validate, y_validate)
        tensor_x = self.embedding.process_x_dataset(x_train)
        tensor_y = self.embedding.process_y_dataset(y_train)

        validation_data = None
        if x_validate is not None:
            tensor_valid_x = self.embedding.process_x_dataset(x_validate)
            tensor_valid_y = self.embedding.process_y_dataset(y_validate)
            validation_data = (tensor_valid_x, tensor_valid_y)

        if fit_kwargs is None:
            fit_kwargs = {}

        if callbacks and 'callbacks' not in fit_kwargs:
            fit_kwargs['callbacks'] = callbacks

        with utils.custom_object_scope():
            return self.tf_model.fit(tensor_x, tensor_y,
                                     validation_data=validation_data,
                                     epochs=epochs,
                                     batch_size=batch_size,
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
        if not kashgari.config.disable_auto_summary:
            self.tf_model.summary()

    def predict(self,
                x_data,
                batch_size=32,
                debug_info=False,
                predict_kwargs: Dict = None):
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            debug_info: Bool, Should print out the logging info.
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            array(s) of predictions.
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        with utils.custom_object_scope():
            if isinstance(x_data, tuple):
                lengths = [len(sen) for sen in x_data[0]]
            else:
                lengths = [len(sen) for sen in x_data]
            tensor = self.embedding.process_x_dataset(x_data)
            pred = self.tf_model.predict(tensor, batch_size=batch_size, **predict_kwargs)
            if self.task == 'scoring':
                t_pred = pred
            else:
                t_pred = pred.argmax(-1)
            res = self.embedding.reverse_numerize_label_sequences(t_pred,
                                                                  lengths)
            if debug_info:
                print('input: {}'.format(tensor))
                print('output: {}'.format(pred))
                print('output argmax: {}'.format(t_pred))
        return res

    def evaluate(self,
                 x_data,
                 y_data,
                 batch_size=None,
                 digits=4,
                 debug_info=False) -> Tuple[float, float, Dict]:
        """
        Evaluate model
        Args:
            x_data:
            y_data:
            batch_size:
            digits:
            debug_info:

        Returns:

        """
        raise NotImplementedError

    def build_model_arc(self):
        raise NotImplementedError

    def save(self, model_path: str):
        """
        Save model
        Args:
            model_path:

        Returns:

        """
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)

        with open(os.path.join(model_path, 'model_info.json'), 'w') as f:
            f.write(json.dumps(self.info(), indent=2, ensure_ascii=True))
            f.close()

        self.tf_model.save_weights(os.path.join(model_path, 'model_weights.h5'))
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))


if __name__ == "__main__":
    from kashgari.tasks.labeling import CNN_LSTM_Model
    from kashgari.corpus import ChineseDailyNerCorpus

    train_x, train_y = ChineseDailyNerCorpus.load_data('valid')

    model = CNN_LSTM_Model()
    model.build_model(train_x[:100], train_y[:100])
    r = model.predict_entities(train_x[:5])
    model.save('./res')
    import pprint

    pprint.pprint(r)
    model.evaluate(train_x[:20], train_y[:20])
    print("Hello world")

    print(model.predict(train_x[:20]))
