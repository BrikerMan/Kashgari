# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abc_model.py
# time: 4:30 下午

from abc import ABC
from typing import List, Dict, Any, Union, Optional

import numpy as np
import tensorflow as tf

import kashgari
from kashgari.embeddings import ABCEmbedding, BareEmbedding
from kashgari.generators import CorpusGenerator, BatchDataSet
from kashgari.layers import ConditionalRandomField
from kashgari.logger import logger
from kashgari.metrics.sequence_labeling import get_entities
from kashgari.metrics.sequence_labeling import sequence_labeling_report
from kashgari.processors import SequenceProcessor
from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.types import TextSamplesVar


class ABCLabelingModel(ABCTaskModel, ABC):
    """
    Abstract Labeling Model
    """

    def __init__(self,
                 embedding: ABCEmbedding = None,
                 sequence_length: int = None,
                 hyper_parameters: Dict[str, Dict[str, Any]] = None):
        """

        Args:
            embedding: embedding object
            sequence_length: target sequence length
            hyper_parameters: hyper_parameters to overwrite
        """
        super(ABCLabelingModel, self).__init__()
        if embedding is None:
            embedding = BareEmbedding()  # type: ignore

        if hyper_parameters is None:
            hyper_parameters = self.default_hyper_parameters()

        self.tf_model: Optional[tf.keras.Model] = None
        self.embedding = embedding
        self.hyper_parameters = hyper_parameters
        self.sequence_length = sequence_length
        self.text_processor: SequenceProcessor = SequenceProcessor()
        self.label_processor: SequenceProcessor = SequenceProcessor(build_in_vocab='labeling',
                                                                    min_count=1,
                                                                    build_vocab_from_labels=True)

        self.crf_layer: Optional[ConditionalRandomField] = None

    def build_model(self,
                    x_data: TextSamplesVar,
                    y_data: TextSamplesVar) -> None:
        """
        Build Model with x_data and y_data

        This function will setup a :class:`CorpusGenerator`,
         then call :meth:`ABCClassificationModel.build_model_gen` for preparing processor and model

        Args:
            x_data:
            y_data:

        Returns:

        """

        train_gen = CorpusGenerator(x_data, y_data)
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
            kwargs: additional params passed to :meth:`tf.keras.Model.predict``.
        """
        if loss is None:
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
            y_train: TextSamplesVar,
            x_validate: TextSamplesVar = None,
            y_validate: TextSamplesVar = None,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            fit_kwargs: Dict = None) -> 'tf.keras.callbacks.History':
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
                See :py:class:`tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to :meth:`tf.keras.Model.fit`

        Returns:
            A :py:class:`tf.keras.callback.History`  object. Its `History.history` attribute is
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
                      callbacks: List['tf.keras.callbacks.Callback'] = None,
                      fit_kwargs: Dict = None) -> 'tf.keras.callbacks.History':
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

        train_set = BatchDataSet(train_sample_gen,
                                 text_processor=self.text_processor,
                                 label_processor=self.label_processor,
                                 segment=self.embedding.segment,
                                 seq_length=self.sequence_length,
                                 max_position=self.embedding.max_position,
                                 batch_size=batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}
        if valid_sample_gen:
            valid_set = BatchDataSet(valid_sample_gen,
                                     text_processor=self.text_processor,
                                     label_processor=self.label_processor,
                                     segment=self.embedding.segment,
                                     seq_length=self.sequence_length,
                                     max_position=self.embedding.max_position,
                                     batch_size=batch_size)
            fit_kwargs['validation_data'] = valid_set.take()
            fit_kwargs['validation_steps'] = len(valid_set)

        for x, y in train_set.take(1):
            logger.debug('fit input shape: {}'.format(np.array(x).shape))
            logger.debug('fit input shape: {}'.format(np.array(y).shape))
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
                predict_kwargs: Dict = None) -> List[List[str]]:
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            truncating: remove values from sequences larger than `model.embedding.sequence_length`
            predict_kwargs: arguments passed to :meth:`tf.keras.Model.predict`

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

            print(self.crf_layer)
            tensor = self.text_processor.transform(x_data,
                                                   segment=self.embedding.segment,
                                                   seq_length=seq_length,
                                                   max_position=self.embedding.max_position)
            logger.debug('predict seq_length: {}, input: {}'.format(seq_length, np.array(tensor).shape))
            pred = self.tf_model.predict(tensor, batch_size=batch_size, verbose=1, **predict_kwargs)
            pred = pred.argmax(-1)

            lengths = [len(sen) for sen in x_data]

            res: List[List[str]] = self.label_processor.inverse_transform(pred,  # type: ignore
                                                                          lengths=lengths)
            logger.debug('predict output: {}'.format(np.array(pred).shape))
            logger.debug('predict output argmax: {}'.format(pred))
        return res

    def predict_entities(self,
                         x_data: TextSamplesVar,
                         batch_size: int = 32,
                         join_chunk: str = ' ',
                         truncating: bool = False,
                         predict_kwargs: Dict = None) -> List[Dict]:
        """Gets entities from sequence.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            truncating: remove values from sequences larger than `model.embedding.sequence_length`
            join_chunk: str or False,
            predict_kwargs: arguments passed to :meth:`tf.keras.Model.predict`

        Returns:
            list: list of entity.
        """
        if isinstance(x_data, tuple):
            text_seq = x_data[0]
        else:
            text_seq = x_data
        res = self.predict(x_data,
                           batch_size=batch_size,
                           truncating=truncating,
                           predict_kwargs=predict_kwargs)
        new_res = [get_entities(seq) for seq in res]
        final_res = []
        for index, seq in enumerate(new_res):
            seq_data = []
            for entity in seq:
                res_entities: List[str] = []
                for i, e in enumerate(text_seq[index][entity[1]:entity[2] + 1]):
                    # Handle bert tokenizer
                    if e.startswith('##') and len(res_entities) > 0:
                        res_entities[-1] += e.replace('##', '')
                    else:
                        res_entities.append(e)
                value: Union[str, List[str]]
                if join_chunk is False:
                    value = res_entities
                else:
                    value = join_chunk.join(res_entities)

                seq_data.append({
                    "entity": entity[0],
                    "start": entity[1],
                    "end": entity[2],
                    "value": value,
                })

            final_res.append({
                'tokenized': x_data[index],
                'labels': seq_data
            })
        return final_res

    def evaluate(self,
                 x_data: TextSamplesVar,
                 y_data: TextSamplesVar,
                 batch_size: int = 32,
                 digits: int = 4,
                 truncating: bool = False) -> Dict:
        """
        Build a text report showing the main labeling metrics.

        Args:
            x_data:
            y_data:
            batch_size:
            digits:
            truncating:

        Returns:
            A report dict

        Example:

            >>> from kashgari.tasks.labeling import BiGRU_Model
            >>> model = BiGRU_Model()
            >>> model.fit(train_x, train_y, valid_x, valid_y)
            >>> report = model.evaluate(test_x, test_y)
                       precision    recall  f1-score   support
                <BLANKLINE>
                      ORG     0.0665    0.1108    0.0831       984
                      LOC     0.1870    0.2086    0.1972      1951
                      PER     0.1685    0.0882    0.1158       884
                <BLANKLINE>
                micro avg     0.1384    0.1555    0.1465      3819
                macro avg     0.1516    0.1555    0.1490      3819
                <BLANKLINE>
            >>> print(report)
                {
                 'f1-score': 0.14895159934887792,
                 'precision': 0.1516294012813676,
                 'recall': 0.15553809897879026,
                 'support': 3819,
                 'detail': {'LOC': {'f1-score': 0.19718992248062014,
                                    'precision': 0.18695452457510336,
                                    'recall': 0.20861096873398258,
                                    'support': 1951},
                            'ORG': {'f1-score': 0.08307926829268293,
                                    'precision': 0.06646341463414634,
                                    'recall': 0.11077235772357724,
                                    'support': 984},
                            'PER': {'f1-score': 0.11581291759465479,
                                    'precision': 0.16846652267818574,
                                    'recall': 0.08823529411764706,
                                    'support': 884}},
                }

        """
        y_pred = self.predict(x_data,
                              batch_size=batch_size,
                              truncating=truncating)
        y_true = [seq[:len(y_pred[index])] for index, seq in enumerate(y_data)]

        new_y_pred = []
        for x in y_pred:
            new_y_pred.append([str(i) for i in x])
        new_y_true = []
        for x in y_true:
            new_y_true.append([str(i) for i in x])

        report = sequence_labeling_report(y_true, y_pred, digits=digits)
        return report


if __name__ == "__main__":
    pass
