# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abc_model.py
# time: 4:30 下午

from abc import ABC
from typing import List, Dict, Any
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.types import TextSamplesVar
from kashgari.generators import CorpusGenerator
from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.processors import SequenceProcessor
from kashgari.generators import BatchDataGenerator


class ABCLabelingModel(ABCTaskModel, ABC):
    def __init__(self,
                 embedding: ABCEmbedding = None,
                 hyper_parameters: Dict[str, Dict[str, Any]] = None,
                 **kwargs):
        super(ABCLabelingModel, self).__init__(embedding=embedding,
                                               hyper_parameters=hyper_parameters,
                                               **kwargs)
        self.default_labeling_processor = SequenceProcessor(vocab_dict_type='labeling',
                                                            min_count=1)

    def fit(self,
            x_train: TextSamplesVar,
            y_train: TextSamplesVar,
            x_validate: TextSamplesVar = None,
            y_validate: TextSamplesVar = None,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List = None,
            fit_kwargs: Dict = None):
        """
        Trains the model for a given number of epochs with given data set list.

        Args:
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
                Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
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
        return self.fit_generator(train_sample_gen=train_gen,
                                  valid_sample_gen=valid_gen,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  fit_kwargs=fit_kwargs)

    def fit_generator(self,
                      train_sample_gen: CorpusGenerator,
                      valid_sample_gen: CorpusGenerator = None,
                      batch_size: int = 64,
                      epochs: int = 5,
                      callbacks: List = None,
                      fit_kwargs: Dict = None):
        """
        Trains the model for a given number of epochs with given data generator.

        Data generator must be the subclass of `CorpusGenerator`

        Args:
            train_sample_gen: train data generator.
            valid_sample_gen: valid data generator.
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
                Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to ``fit()`` function from
                ``tensorflow.keras.Model`` - https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        self.build_model(train_sample_gen)
        self.tf_model.summary()

        train_gen = BatchDataGenerator(train_sample_gen,
                                       text_processor=self.embedding.text_processor,
                                       label_processor=self.embedding.label_processor,
                                       segment=self.embedding.segment,
                                       seq_length=self.embedding.sequence_length,
                                       batch_size=batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}
        if valid_sample_gen:
            valid_gen = BatchDataGenerator(valid_sample_gen,
                                           text_processor=self.embedding.text_processor,
                                           label_processor=self.embedding.label_processor,
                                           segment=self.embedding.segment,
                                           seq_length=self.embedding.sequence_length,
                                           batch_size=batch_size)
            fit_kwargs['validation_data'] = valid_gen.generator()
            fit_kwargs['validation_steps'] = valid_gen.steps
        if callbacks:
            fit_kwargs['callbacks'] = callbacks

        return self.tf_model.fit(train_gen.generator(),
                                 steps_per_epoch=train_gen.steps,
                                 epochs=epochs,
                                 callbacks=callbacks)


if __name__ == "__main__":
    pass
