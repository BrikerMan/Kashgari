# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 2019-05-20 13:07


from typing import Dict, Any, List, Optional

from tensorflow import keras

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
            self.embedding = BareEmbedding()
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

    def fit(self,
            x_train: List[List[str]],
            y_train: List[List[str]],
            x_validate: List[List[str]] = None,
            y_validate: List[List[str]] = None,
            batch_size: int = 64,
            epochs: int = 5,
            fit_kwargs: Dict = None,
            **kwargs):
        """
        Trains the model for a given number of epochs (iterations on a dataset).

        Args:
            x_train: Array of training data
            y_train: Array of training data
            x_validate: Array of validation data
            y_validate: Array of validation data
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Integer. Number of epochs to train the model. default 5.
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit_generator()`` function from
                ``tensorflow.keras.Model`` -
                https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator
            **kwargs:

        Returns:

        """
        if self.embedding.token_count == 0:
            if x_validate is not None:
                x_all = (x_train + x_validate).copy()
                y_all = (y_train + y_validate).copy()
            else:
                x_all = x_train.copy()
                y_all = y_train.copy()
            self.embedding.prepare_for_labeling(x_all, y_all)

        if self.tf_model is None:
            self.prepare_model_arc()
            self.compile_model()

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

    def prepare_model_arc(self):
        raise NotImplementedError


if __name__ == "__main__":
    print("Hello world")
