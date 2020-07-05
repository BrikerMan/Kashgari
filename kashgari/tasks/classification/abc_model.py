# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_model.py
# time: 4:05 下午

import random
from abc import ABC
import numpy as np
from typing import List, Dict, Any, Union

from sklearn import metrics as sklearn_metrics
from tensorflow import keras

import kashgari
from kashgari.embeddings import ABCEmbedding, BareEmbedding
from kashgari.generators import BatchDataSet, CorpusGenerator
from kashgari.layers import L
from kashgari.logger import logger
from kashgari.metrics.multi_label_classification import multi_label_classification_report
from kashgari.processors import ABCProcessor
from kashgari.processors import ClassificationProcessor
from kashgari.processors import SequenceProcessor
from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.types import TextSamplesVar, ClassificationLabelVar, MultiLabelClassificationLabelVar


class ABCClassificationModel(ABCTaskModel, ABC):
    """
    Abstract Classification Model
    """

    __task__ = 'classification'

    def to_dict(self) -> Dict:
        info = super(ABCClassificationModel, self).to_dict()
        info['config']['multi_label'] = self.multi_label
        return info

    def __init__(self,
                 embedding: ABCEmbedding = None,
                 *,
                 sequence_length: int = None,
                 hyper_parameters: Dict[str, Dict[str, Any]] = None,
                 multi_label: bool = False,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None):
        """

        Args:
            embedding: embedding object
            sequence_length: target sequence length
            hyper_parameters: hyper_parameters to overwrite
            multi_label: is multi-label classification
            text_processor: text processor
            label_processor: label processor
        """
        super(ABCClassificationModel, self).__init__()
        if embedding is None:
            embedding = BareEmbedding()  # type: ignore

        if hyper_parameters is None:
            hyper_parameters = self.default_hyper_parameters()

        if text_processor is None:
            text_processor = SequenceProcessor()

        if label_processor is None:
            label_processor = ClassificationProcessor(multi_label=multi_label)

        self.tf_model: keras.Model = None
        self.embedding = embedding
        self.hyper_parameters = hyper_parameters
        self.sequence_length = sequence_length
        self.multi_label = multi_label

        self.text_processor = text_processor
        self.label_processor = label_processor

    def _activation_layer(self) -> L.Layer:
        if self.multi_label:
            return L.Activation('sigmoid')
        else:
            return L.Activation('softmax')

    def build_model(self,
                    x_train: TextSamplesVar,
                    y_train: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar]) -> None:
        """
        Build Model with x_data and y_data

        This function will setup a :class:`CorpusGenerator`,
         then call py:meth:`ABCClassificationModel.build_model_gen` for preparing processor and model

        Args:
            x_train:
            y_train:

        Returns:

        """

        train_gen = CorpusGenerator(x_train, y_train)
        self.build_model_generator([train_gen])

    def build_model_generator(self,
                              generators: List[CorpusGenerator]) -> None:
        if not self.text_processor.vocab2idx:
            self.text_processor.build_vocab_generator(generators)
        self.label_processor.build_vocab_generator(generators)
        self.embedding.setup_text_processor(self.text_processor)

        if self.sequence_length is None:
            self.sequence_length = self.embedding.get_seq_length_from_corpus(generators)

        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

    def build_model_arc(self) -> None:
        raise NotImplementedError

    def compile_model(self,
                      loss: Any = None,
                      optimizer: Any = None,
                      metrics: Any = None,
                      **kwargs: Any) -> None:
        """
        Configures the model for training.
        call :meth:`tf.keras.Model.predict` to compile model with custom loss, optimizer and metrics

        Examples:

            >>> model = BiLSTM_Model()
            # Build model with corpus
            >>> model.build_model(train_x, train_y)
            # Compile model with custom loss, optimizer and metrics
            >>> model.compile(loss='categorical_crossentropy', optimizer='rsm', metrics = ['accuracy'])

        Args:
            loss: name of objective function, objective function or ``tf.keras.losses.Loss`` instance.
            optimizer: name of optimizer or optimizer instance.
            metrics (object): List of metrics to be evaluated by the model during training and testing.
            **kwargs: additional params passed to :meth:`tf.keras.Model.predict``.
        """
        if loss is None:
            if self.multi_label:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
        if optimizer is None:
            optimizer = 'adam'
        if metrics is None:
            metrics = ['accuracy']

        self.tf_model.compile(loss=loss,
                              optimizer=optimizer,
                              metrics=metrics,
                              **kwargs)

    def fit(self,
            x_train: TextSamplesVar,
            y_train: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar],
            x_validate: TextSamplesVar = None,
            y_validate: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar] = None,
            *,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List['keras.callbacks.Callback'] = None,
            fit_kwargs: Dict = None) -> 'keras.callbacks.History':
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
            callbacks: List of `tf.keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See :class:`tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to :meth:`tf.keras.Model.fit`

        Returns:
            A :class:`tf.keras.callback.History`  object. Its `History.history` attribute is
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
                      *,
                      batch_size: int = 64,
                      epochs: int = 5,
                      callbacks: List['keras.callbacks.Callback'] = None,
                      fit_kwargs: Dict = None) -> 'keras.callbacks.History':
        """
        Trains the model for a given number of epochs with given data generator.

        Data generator must be the subclass of `CorpusGenerator`

        Args:
            train_sample_gen: train data generator.
            valid_sample_gen: valid data generator.
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
            callbacks: List of `tf.keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to :meth:`tf.keras.Model.fit`

        Returns:
            A :py:class:`tf.keras.callback.History`  object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        self.build_model_generator([g for g in [train_sample_gen, valid_sample_gen] if g])

        model_summary = []
        self.tf_model.summary(print_fn=lambda x: model_summary.append(x))
        logger.debug('\n'.join(model_summary))

        train_set = BatchDataSet(train_sample_gen,
                                 text_processor=self.text_processor,
                                 label_processor=self.label_processor,
                                 segment=self.embedding.segment,
                                 seq_length=self.sequence_length,
                                 batch_size=batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}

        if valid_sample_gen:
            valid_gen = BatchDataSet(valid_sample_gen,
                                     text_processor=self.text_processor,
                                     label_processor=self.label_processor,
                                     segment=self.embedding.segment,
                                     seq_length=self.sequence_length,
                                     batch_size=batch_size)
            fit_kwargs['validation_data'] = valid_gen.take()
            fit_kwargs['validation_steps'] = len(valid_gen)

        return self.tf_model.fit(train_set.take(),
                                 steps_per_epoch=len(train_set),
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 **fit_kwargs)

    def predict(self,
                x_data: TextSamplesVar,
                *,
                batch_size: int = 32,
                truncating: bool = False,
                multi_label_threshold: float = 0.5,
                predict_kwargs: Dict = None) -> Union[ClassificationLabelVar, MultiLabelClassificationLabelVar]:
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            truncating: remove values from sequences larger than `model.embedding.sequence_length`
            multi_label_threshold:
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            array(s) of predictions.
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        with kashgari.utils.custom_object_scope():
            if truncating:
                seq_length = self.sequence_length
            else:
                seq_length = None
            tensor = self.text_processor.transform(x_data,
                                                   segment=self.embedding.segment,
                                                   seq_length=seq_length,
                                                   max_position=self.embedding.max_position)
            logger.debug(f'predict input shape {np.array(tensor).shape} x: \n{tensor}')
            pred = self.tf_model.predict(tensor, batch_size=batch_size, **predict_kwargs)
            logger.debug(f'predict output shape {pred.shape}')
            if self.multi_label:
                multi_label_binarizer = self.label_processor.multi_label_binarizer  # type: ignore
                res = multi_label_binarizer.inverse_transform(pred,
                                                              threshold=multi_label_threshold)
            else:
                pred_argmax = pred.argmax(-1)
                lengths = [len(sen) for sen in x_data]
                res = self.label_processor.inverse_transform(pred_argmax,
                                                             lengths=lengths)
                logger.debug(f'predict output argmax: {pred_argmax}')

        return res

    def evaluate(self,  # type: ignore[override]
                 x_data: TextSamplesVar,
                 y_data: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar],
                 *,
                 batch_size: int = 32,
                 digits: int = 4,
                 multi_label_threshold: float = 0.5,
                 truncating: bool = False,) -> Dict:
        y_pred = self.predict(x_data,
                              batch_size=batch_size,
                              truncating=truncating,
                              multi_label_threshold=multi_label_threshold)

        if self.multi_label:
            report = multi_label_classification_report(y_data,  # type: ignore
                                                       y_pred,  # type: ignore
                                                       binarizer=self.label_processor.multi_label_binarizer)  # type: ignore

        else:
            original_report = sklearn_metrics.classification_report(y_data,
                                                                    y_pred,
                                                                    output_dict=True,
                                                                    digits=digits)
            print(sklearn_metrics.classification_report(y_data,
                                                        y_pred,
                                                        output_dict=False,
                                                        digits=digits))
            report = {
                'detail': original_report,
                **original_report['weighted avg']
            }
        return report


if __name__ == "__main__":
    pass
