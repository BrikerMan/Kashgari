# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: generator.py
# time: 4:53 下午

from abc import ABC
from typing import Iterable, Iterator, TYPE_CHECKING
from typing import List, Any, Tuple

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from kashgari.processors.abc_processor import ABCProcessor


class ABCGenerator(Iterable, ABC):
    def __init__(self, buffer_size: int = 2000) -> None:
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def sample(self) -> Iterator[Tuple[Any, Any]]:
        buffer, is_full = [], False
        for sample in self:
            buffer.append(sample)
            if is_full:
                i = np.random.randint(len(buffer))
                yield buffer.pop(i)
            elif len(buffer) == self.buffer_size:
                is_full = True
        while buffer:
            i = np.random.randint(len(buffer))
            yield buffer.pop(i)


class CorpusGenerator(ABCGenerator):

    def __init__(self,
                 x_data: List,
                 y_data: List,
                 *,
                 buffer_size: int = 2000) -> None:
        super(CorpusGenerator, self).__init__(buffer_size=buffer_size)
        self.x_data = x_data
        self.y_data = y_data
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        for i in range(len(self.x_data)):
            yield self.x_data[i], self.y_data[i]

    def __len__(self) -> int:
        return len(self.x_data)


class BatchDataSet(Iterable):
    def __init__(self,
                 corpus: CorpusGenerator,
                 *,
                 text_processor: 'ABCProcessor',
                 label_processor: 'ABCProcessor',
                 seq_length: int = None,
                 max_position: int = None,
                 segment: bool = False,
                 batch_size: int = 64) -> None:
        self.corpus = corpus
        self.text_processor = text_processor
        self.label_processor = label_processor

        self.seq_length = seq_length
        self.max_position = max_position
        self.segment = segment

        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(len(self.corpus) // self.batch_size, 1)

    def __iter__(self) -> Iterator:
        batch_x, batch_y = [], []
        for x, y in self.corpus.sample():
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == self.batch_size:
                x_tensor = self.text_processor.transform(batch_x,
                                                         seq_length=self.seq_length,
                                                         max_position=self.max_position,
                                                         segment=self.segment)
                y_tensor = self.label_processor.transform(batch_y,
                                                          seq_length=self.seq_length,
                                                          max_position=self.max_position)
                yield x_tensor, y_tensor
                batch_x, batch_y = [], []

    def take(self, batch_count: int = None) -> Any:
        """
        take batches from the dataset

        Args:
            batch_count: number of batch count, iterate forever when batch_count is None.
        """
        i = 0
        should_continue = True
        while should_continue:
            for batch_x, batch_y in self.__iter__():
                if batch_count is None or i < batch_count:
                    i += 1
                    yield batch_x, batch_y
                if batch_count and i >= batch_count:
                    should_continue = False
                    break

        # x_shape = self.text_processor.get_tensor_shape(self.batch_size, self.seq_length)
        # y_shape = self.label_processor.get_tensor_shape(self.batch_size, self.seq_length)
        # dataset = tf.data.Dataset.from_generator(self.__iter__,
        #                                          output_types=(tf.int64, tf.int64),
        #                                          output_shapes=(x_shape, y_shape))
        # dataset = dataset.repeat()
        # dataset = dataset.prefetch(50)
        # if batch_count is None:
        #     batch_count = len(self)
        # return dataset.take(batch_count)


class Seq2SeqDataSet(Iterable):
    def __init__(self,
                 corpus: CorpusGenerator,
                 *,
                 batch_size: int = 64,
                 encoder_processor: 'ABCProcessor',
                 decoder_processor: 'ABCProcessor',
                 encoder_seq_length: int = None,
                 decoder_seq_length: int = None,
                 encoder_segment: bool = False,
                 decoder_segment: bool = False):
        self.corpus = corpus

        self.encoder_processor = encoder_processor
        self.decoder_processor = decoder_processor

        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length

        self.encoder_segment = encoder_segment
        self.decoder_segment = decoder_segment

        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(len(self.corpus) // self.batch_size, 1)

    def __iter__(self) -> Iterator:
        batch_x, batch_y = [], []
        for x, y in self.corpus.sample():
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == self.batch_size:
                x_tensor = self.encoder_processor.transform(batch_x,
                                                            seq_length=self.encoder_seq_length,
                                                            segment=self.encoder_segment)
                y_tensor = self.decoder_processor.transform(batch_y,
                                                            seq_length=self.decoder_seq_length,
                                                            segment=self.encoder_segment)
                yield x_tensor, y_tensor
                batch_x, batch_y = [], []

    def take(self, batch_count: int = None) -> Any:
        x_shape = [self.batch_size, self.encoder_seq_length]
        y_shape = [self.batch_size, self.decoder_seq_length]
        dataset = tf.data.Dataset.from_generator(self.__iter__,
                                                 output_types=(tf.int64, tf.int64),
                                                 output_shapes=(x_shape, y_shape))
        dataset = dataset.repeat()
        dataset = dataset.prefetch(50)
        if batch_count is None:
            batch_count = len(self)
        return dataset.take(batch_count)
