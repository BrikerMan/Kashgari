# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-20 11:13

import logging

from tensorflow import keras
from kashgari.embeddings import BareEmbedding
from typing import Dict, Any, List, Optional
L = keras.layers


class BLSTMModel(object):
    """Bidirectional LSTM Sequence Labeling Model"""
    __architect_name__ = 'BLSTMModel'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_blstm': {
                'units': 256,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {
                'activation': 'softmax'
            }
        }

    def __init__(self,
                 embedding: Optional[BareEmbedding] = None,
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

        self.tf_model: keras.Model = None
        self.hyper_parameters = self.get_default_hyper_parameters()
        self._label2idx = {}
        self._idx2label = {}
        self.model_info = {}
        self.pre_processor = self.embedding.processor

        if hyper_parameters:
            self.hyper_parameters.update(hyper_parameters)

    def prepare_model_arc(self):


        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters

        embed_model = self.embedding.embed_model
        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']), name='layer_blstm')
        layer_dropout = L.Dropout(**config['layer_dropout'], name='layer_dropout')
        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim, **config['layer_time_distributed']),
                                                   name='layer_time_distributed')

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dropout(tensor)
        output_tensor = layer_time_distributed(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self,
                      loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy']):
        self.tf_model.compile(loss=loss,
                              optimizer=optimizer,
                              metrics=metrics)
        self.tf_model.summary()

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


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.pre_processors import PreProcessor

    x, y = ChineseDailyNerCorpus.load_data()

    # Old fashion model
    old_fashion_model = BLSTMModel()
    old_fashion_model.fit(x, y)

    # Model For pros 1
    embedding = BareEmbedding(sequence_length=20)
    embedding.prepare_for_labeling(x, y)
    embedding.processor.save_dicts('./cached_processor')

    print(embedding.embed_model.summary())
    labeling1 = BLSTMModel(embedding=embedding)
    labeling1.prepare_model_arc()
    labeling1.compile_model()

    # Model For pros 1
    processor = PreProcessor.load_cached_processor('./cached_processor')
    embedding = BareEmbedding(sequence_length='variable', processor=processor)

    labeling2 = BLSTMModel(embedding=embedding)
    labeling2.prepare_model_arc()
    labeling2.compile_model()

    hyper_parameters = BLSTMModel.get_default_hyper_parameters()
    hyper_parameters['layer_blstm']['units'] = 12
    labeling_model = BLSTMModel(hyper_parameters=hyper_parameters)
    labeling_model.fit(x, y)



