# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_task_model.py
# time: 1:43 下午

from abc import ABC
from typing import List, Dict, Any
from kashgari.embeddings import WordEmbedding
from kashgari.embeddings import BareEmbedding
from kashgari.generators import CorpusGenerator

from tensorflow import keras


class ABCTaskModel(ABC):

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

        You could easily change model's hyper-parameters. For example, change the LSTM unit in BiLSTM_Model from 128 to 32.

        Example::

            from kashgari.tasks.classification import BiLSTM_Model

            hyper = BiLSTM_Model.get_default_hyper_parameters()
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
            self.label_processor.sequence_length = self.text_processor.sequence_length
        self.embedding.build_generator(train_gen)
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


if __name__ == "__main__":
    pass
