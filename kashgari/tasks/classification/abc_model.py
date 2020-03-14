# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_model.py
# time: 4:05 下午

from abc import ABC
from typing import List, Dict
from kashgari.embeddings import WordEmbedding
from kashgari.typing import NumSamplesListVar, TextSamplesVar
from kashgari.generators import CorpusGenerator

from tensorflow import keras
from tensorflow.keras.callbacks import Callback


class ABCClassificationModel(ABC):
    def __init__(self, embedding: WordEmbedding = None):
        self.embedding = embedding

        self.tf_model: keras.Model = None

    @property
    def text_processor(self):
        return self.embedding.text_processor

    @property
    def label_processor(self):
        return self.embedding.label_processor

    def build_model(self,
                    train_gen: CorpusGenerator):
        if self.embedding.label_processor is None:
            from kashgari.processor.class_processor import ClassificationProcessor
            self.embedding.label_processor = ClassificationProcessor()
        self.embedding.build(train_gen)
        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

    def build_model_arc(self):
        raise NotImplementedError

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
        # if not kashgari.config.disable_auto_summary:
        #     self.tf_model.summary()

    def fit(self,
            x_train: NumSamplesListVar,
            y_train: NumSamplesListVar,
            x_validate: NumSamplesListVar = None,
            y_validate: NumSamplesListVar = None,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List[Callback] = None,
            fit_kwargs: Dict = None, ):
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
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit()`` function from
                ``tensorflow.keras.Model`` - https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        train_gen = CorpusGenerator(x_train, y_train)
        if x_validate is not None:
            valid_gen = CorpusGenerator(x_validate, y_validate)
        else:
            valid_gen = None
        return self.fit_generator(train_gen=train_gen,
                                  valid_gen=valid_gen,
                                  batch_size=batch_size,
                                  epochs=epochs)

    def fit_generator(self,
                      train_gen: CorpusGenerator,
                      valid_gen: CorpusGenerator = None,
                      batch_size: int = 64,
                      epochs: int = 5):
        self.build_model(train_gen)
        self.tf_model.summary()
        from kashgari.generators import BatchDataGenerator
        train_gen_batch = BatchDataGenerator(train_gen,
                                             self.embedding.text_processor,
                                             self.embedding.label_processor,
                                             batch_size=batch_size)

        return self.tf_model.fit(train_gen_batch,
                                 steps_per_epoch=train_gen_batch.steps,
                                 epochs=epochs)


if __name__ == "__main__":
    pass
