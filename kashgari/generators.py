# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: generator.py
# time: 4:53 下午

from abc import ABC
import random
from typing import List
from typing import Iterable


class ABCGenerator(Iterable, ABC):

    @property
    def steps(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class CorpusGenerator(ABCGenerator):

    def __init__(self, x_data: List, y_data: List, shuffle=True):
        self.x_data = x_data
        self.y_data = y_data

        self._index_list = list(range(len(self.x_data)))

        if shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self._index_list)

    def __iter__(self):
        for i in self._index_list:
            yield self.x_data[i], self.y_data[i]

    @property
    def steps(self) -> int:
        return len(self.x_data)


class BatchDataGenerator(Iterable):
    def __init__(self,
                 corpus,
                 text_processor,
                 label_processor,
                 seq_length: int = None,
                 max_position: int = None,
                 segment: bool = False,
                 batch_size=64,
                 buffer_size=None):
        self.corpus = corpus
        self.text_processor = text_processor
        self.label_processor = label_processor

        self.seq_length = seq_length
        self.max_position = max_position
        self.segment = segment

        self.batch_size = batch_size
        if buffer_size is None:
            self.buffer_size = min(self.batch_size * 200, self.corpus.steps)
        else:
            self.buffer_size = min(buffer_size, self.corpus.steps)
        self.forever = True

    @property
    def steps(self) -> int:
        return max(self.corpus.steps // self.batch_size, 1)

    def __iter__(self):
        from kashgari.utils import unison_shuffled_copies
        while True:
            x_buffer, y_buffer = [], []
            for x, y in self.corpus:
                x_buffer.append(x)
                y_buffer.append(y)

                if len(x_buffer) == self.buffer_size:
                    x_buffer, y_buffer = unison_shuffled_copies(x_buffer, y_buffer)

                    x_set, y_set = [], []
                    for x, y in zip(x_buffer, y_buffer):
                        x_set.append(x)
                        y_set.append(y)
                        if len(x_set) == self.batch_size:
                            x_tensor = self.text_processor.numerize_samples(x_set,
                                                                            seq_length=self.seq_length,
                                                                            max_position=self.max_position,
                                                                            segment=self.segment)
                            y_tensor = self.label_processor.numerize_samples(y_set, seq_length=self.seq_length,
                                                                             one_hot=True)
                            yield x_tensor, y_tensor
                            x_set, y_set = [], []
                        x_buffer, y_buffer = [], []
            # final step
            if x_set:
                x_tensor = self.text_processor.numerize_samples(x_set, seq_length=self.seq_length, segment=self.segment)
                y_tensor = self.label_processor.numerize_samples(y_set, seq_length=self.seq_length, one_hot=True)
                yield x_tensor, y_tensor
            if not self.forever:
                break

    def generator(self):
        for item in self:
            yield item
