# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_task_model.py
# time: 1:43 下午

import json
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List

from tensorflow import keras

from kashgari.embeddings import BareEmbedding
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor


class ABCTaskModel(ABC):

    def info(self) -> Dict:
        import kashgari
        import tensorflow as tf
        model_json_str = self.tf_model.to_json()  # type: ignore

        return {
            'config': {
                'hyper_parameters': self.hyper_parameters,  # type: ignore
            },
            'tf_model': json.loads(model_json_str),
            'embedding': self.embedding.info(),  # type: ignore
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'tf_version': tf.__version__,  # type: ignore
            'kashgari_version': kashgari.__version__
        }

    # def __init__(self,
    #              embedding: ABCEmbedding = None,
    #              *,
    #              sequence_length: int = None,
    #              hyper_parameters: Dict[str, Dict[str, Any]] = None,
    #              **kwargs: Any) -> None:
    #     self.tf_model: keras.Model = None
    #     self.embedding: ABCEmbedding
    #     if embedding is None:
    #         self.embedding = BareEmbedding()  # type: ignore
    #     else:
    #         self.embedding = embedding
    #
    #     if sequence_length and sequence_length != self.embedding.sequence_length:
    #         if self.embedding.sequence_length is None:
    #             self.embedding.set_sequence_length(sequence_length)
    #         else:
    #             raise ValueError("Already setup embedding's sequence_length, if need to change sequence length, call "
    #                              "`model.embedding.set_sequence_length(sequence_length)`")
    #
    #     self.hyper_parameters = self.default_hyper_parameters().copy()
    #     if hyper_parameters:
    #         self.hyper_parameters.update(hyper_parameters)
    #     self.default_labeling_processor: ABCProcessor = None

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        The default hyper parameters of the model dict, **all models must implement this function.**

        You could easily change model's hyper-parameters.

        For example, change the LSTM unit in BiLSTM_Model from 128 to 32.

            >>> from kashgari.tasks.classification import BiLSTM_Model
            >>> hyper = BiLSTM_Model.default_hyper_parameters()
            >>> print(hyper)
            {'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_output': {}}
            >>> hyper['layer_bi_lstm']['units'] = 32
            >>> model = BiLSTM_Model(hyper_parameters=hyper)

        Returns:
            hyper params dict
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self,
                    x_train: Any,
                    y_train: Any) -> None:
        raise NotImplementedError

    # def build_model_arc(self) -> None:
    #     """
    #     Build model architect, **all models must implement this function.**
    #     Returns:
    #
    #     """
    #     raise NotADirectoryError

    # def build_model_generator(self,
    #                           train_gen: CorpusGenerator) -> None:
    #     """
    #     Build model with a generator, This function will do:
    #
    #     1. setup processor's vocab if the vocab is empty.
    #     2. calculate the sequence length if `sequence_length` is None.
    #     3. build up model architect.
    #     4. compile the ``tf_model`` with default loss, optimizer and metrics.
    #
    #     Args:
    #         train_gen: train data generator
    #
    #     """
    #     if self.embedding.label_processor is None:
    #         if self.default_labeling_processor is None:
    #             raise ValueError('Need to set default_labeling_processor')
    #         self.embedding.label_processor = self.default_labeling_processor
    #     self.embedding.build_with_generator(train_gen)
    #     self.embedding.calculate_sequence_length_if_needs(train_gen)
    #     if self.tf_model is None:
    #         self.build_model_arc()
    #         self.compile_model()
    #         if self.tf_model is not None:
    #             self.tf_model.summary()

    # def compile_model(self,
    #                   loss: Any = None,
    #                   optimizer: Any = None,
    #                   metrics: Any = None,
    #                   **kwargs: Any) -> None:
    #     """
    #     Configures the model for training.
    #     call :meth:`tf.keras.Model.predict` to compile model with custom loss, optimizer and metrics
    #
    #     Examples:
    #
    #         >>> model = BiLSTM_Model()
    #         # Build model with corpus
    #         >>> model.build_model(train_x, train_y)
    #         # Compile model with custom loss, optimizer and metrics
    #         >>> model.compile(loss='categorical_crossentropy', optimizer='rsm', metrics = ['accuracy'])
    #
    #     Args:
    #         loss: name of objective function, objective function or ``tf.keras.losses.Loss`` instance.
    #         optimizer: name of optimizer or optimizer instance.
    #         metrics: List of metrics to be evaluated by the model during training and testing.
    #         **kwargs: additional params passed to :meth:`tf.keras.Model.predict``.
    #     """
    #     if kwargs.get('loss') is None:
    #         kwargs['loss'] = 'categorical_crossentropy'
    #     if kwargs.get('optimizer') is None:
    #         kwargs['optimizer'] = 'adam'
    #     if kwargs.get('metrics') is None:
    #         kwargs['metrics'] = ['accuracy']
    #
    #     self.tf_model.compile(**kwargs)

    def save(self, model_path: str) -> str:
        """
        Save model
        Args:
            model_path:
        """
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
        model_path = os.path.abspath(model_path)

        with open(os.path.join(model_path, 'model_info.json'), 'w') as f:
            f.write(json.dumps(self.info(), indent=2, ensure_ascii=True))
            f.close()

        self.tf_model.save_weights(os.path.join(model_path, 'model_weights.h5'))  # type: ignore
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))
        return model_path

    # def predict(self,
    #             x_data: Any,
    #             *,
    #             batch_size: int = 32,
    #             truncating: bool = False,
    #             debug_info: bool = False,
    #             predict_kwargs: Dict = None,
    #             **kwargs: Any) -> List[Union[List[str], str]]:
    #     raise NotImplementedError
    #
    # def evaluate(self,
    #              x_data: Any,
    #              y_data: Any,
    #              *,
    #              batch_size: int = 32,
    #              digits: int = 4,
    #              truncating: bool = False,
    #              debug_info: bool = False,
    #              **kwargs: Dict) -> Dict:
    #     raise NotImplementedError


if __name__ == "__main__":
    pass
