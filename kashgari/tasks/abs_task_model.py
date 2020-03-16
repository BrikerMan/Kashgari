# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_task_model.py
# time: 1:43 下午

import os
import json
import pathlib
import logging

from abc import ABC
from typing import Dict, Any

import kashgari
from kashgari.embeddings import WordEmbedding
from kashgari.embeddings import BareEmbedding
from kashgari.generators import CorpusGenerator

from tensorflow import keras


class ABCTaskModel(ABC):

    def info(self) -> Dict:
        import kashgari
        import tensorflow as tf
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

    def __init__(self,
                 embedding: WordEmbedding = None,
                 hyper_parameters: Dict[str, Dict[str, Any]] = None,
                 **kwargs):
        self.tf_model: keras.Model = None
        if embedding is None:
            self.embedding = BareEmbedding()
        else:
            self.embedding = embedding
        self.hyper_parameters = self.default_hyper_parameters().copy()
        if hyper_parameters:
            self.hyper_parameters.update(hyper_parameters)
        self.default_labeling_processor = None

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        The default hyper parameters of the model dict, **all models must implement this function.**

        You could easily change model's hyper-parameters.

        For example, change the LSTM unit in BiLSTM_Model from 128 to 32.::

            from kashgari.tasks.classification import BiLSTM_Model

            hyper = BiLSTM_Model.default_hyper_parameters()
            print(hyper)
            # {'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_dense': {'activation': 'softmax'}}

            hyper['layer_bi_lstm']['units'] = 32
            model = BiLSTM_Model(hyper_parameters=hyper)

        Returns:
            hyper params dict
        """
        raise NotImplementedError

    @property
    def text_processor(self):
        return self.embedding.text_processor

    @property
    def label_processor(self):
        return self.embedding.label_processor

    def build_model(self,
                    train_gen: CorpusGenerator):
        """
        Build model function, will be
        Args:
            train_gen:

        Returns:

        """
        if self.embedding.label_processor is None:
            if self.default_labeling_processor is None:
                raise ValueError('Need to set default_labeling_processor')
            self.embedding.label_processor = self.default_labeling_processor
        self.embedding.build_with_generator(train_gen)
        self.embedding.calculate_sequence_length_if_needs(train_gen)
        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

    def build_model_arc(self):
        """
        Build model architect, **all models must implement this function.**
        Returns:

        """
        raise NotADirectoryError

    def compile_model(self, **kwargs):
        """Configures the model for training.

        Using ``compile()`` function of ``tf.keras.Model`` -
        https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#compile

        Args:
            **kwargs: arguments passed to ``compile()`` function of ``tf.keras.Model``. Default values:
                `loss = categorical_crossentropy`,
                `optimizer = adam`,
                `metrics = ['accuracy']`.
        """
        if kwargs.get('loss') is None:
            kwargs['loss'] = 'categorical_crossentropy'
        if kwargs.get('optimizer') is None:
            kwargs['optimizer'] = 'adam'
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = ['accuracy']

        self.tf_model.compile(**kwargs)

    def save(self, model_path: str):
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

        self.tf_model.save_weights(os.path.join(model_path, 'model_weights.h5'))
        logging.info('model saved to {}'.format(os.path.abspath(model_path)))
        return model_path

    def predict(self,
                x_data,
                batch_size=32,
                debug_info=False,
                predict_kwargs: Dict = None,
                **kwargs):
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
        with kashgari.utils.custom_object_scope():
            tensor = self.embedding.text_processor.numerize_samples(x_data)
            pred = self.tf_model.predict(tensor, batch_size=batch_size, **predict_kwargs)
            pred = pred.argmax(-1)

            res = self.embedding.label_processor.reverse_numerize(pred)
            if debug_info:
                logging.info('input: {}'.format(tensor))
                logging.info('output: {}'.format(pred))
                logging.info('output argmax: {}'.format(pred.argmax(-1)))
        return res


if __name__ == "__main__":
    pass
