# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: generator.py
# time: 4:53 下午

from abc import ABC
import random
import numpy as np
from typing import List, Any, Tuple
from typing import Iterable


class ABCGenerator(Iterable, ABC):
    def __init__(self, buffer_size=2000):
        self.buffer_size = buffer_size

    def __iter__(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def generator(self):
        """
        return a generator with shuffle
        """
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

    def __init__(self, x_data: List, y_data: List, buffer_size=2000):
        super(CorpusGenerator, self).__init__(buffer_size=buffer_size)
        self.x_data = x_data
        self.y_data = y_data
        self.buffer_size = buffer_size

    def __iter__(self) -> Tuple[Any, Any]:
        for i in range(len(self.x_data)):
            yield self.x_data[i], self.y_data[i]

    def __len__(self):
        return len(self.x_data)


class BatchDataGenerator(Iterable):
    def __init__(self,
                 corpus: CorpusGenerator,
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
            self.buffer_size = min(self.batch_size * 200, len(self.corpus))
        else:
            self.buffer_size = min(buffer_size, len(self.corpus))
        self.forever = True

    def __len__(self) -> int:
        return max(len(self.corpus) // self.batch_size, 1)

    def __iter__(self):
        while True:
            batch_x, batch_y = [], []
            for x, y in self.corpus.generator():
                batch_x.append(x)
                batch_y.append(y)
                if len(batch_x) == self.batch_size:
                    x_tensor = self.text_processor.numerize_samples(batch_x,
                                                                    seq_length=self.seq_length,
                                                                    max_position=self.max_position,
                                                                    segment=self.segment)
                    y_tensor = self.label_processor.numerize_samples(batch_y,
                                                                     seq_length=self.seq_length,
                                                                     one_hot=True)
                    yield x_tensor, y_tensor
                    batch_x, batch_y = [], []
            if batch_x:
                x_tensor = self.text_processor.numerize_samples(batch_x,
                                                                seq_length=self.seq_length,
                                                                max_position=self.max_position,
                                                                segment=self.segment)
                y_tensor = self.label_processor.numerize_samples(batch_y,
                                                                 seq_length=self.seq_length,
                                                                 one_hot=True)
                yield x_tensor, y_tensor

            if not self.forever:
                break

    def generator(self):
        for item in self:
            yield item
