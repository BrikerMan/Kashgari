# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_model.py
# time: 4:05 下午

from abc import ABC
from kashgari.embeddings import WordEmbedding
from kashgari.typing import NumSamplesListVar, TextSamplesVar
from kashgari.generator import CorpusGenerator

from tensorflow import keras


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
            epochs: int = 5):
        train_gen = CorpusGenerator(x_train, y_train)
        if x_validate is not None:
            valid_gen = CorpusGenerator(x_validate, y_validate)
        else:
            valid_gen = None
        self.fit_generator(train_gen=train_gen,
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
        from kashgari.generator import BatchDataGenerator
        train_gen_batch = BatchDataGenerator(train_gen,
                                             self.embedding.text_processor,
                                             self.embedding.label_processor,
                                             batch_size=batch_size)

        self.tf_model.fit(train_gen_batch, steps_per_epoch=train_gen_batch.steps, epochs=epochs)


if __name__ == "__main__":
    pass
