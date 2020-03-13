# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: utils.py
# time: 12:39 下午

import random
from types import TracebackType
from typing import List, Generator, Type, Optional


def get_list_subset(target: List, index_list: List[int]) -> List:
    return [target[i] for i in index_list if i < len(target)]


class CorpusGenerator(Generator):

    def reset(self):
        self._current_index = 0

    def throw(self,
              typ: Type[BaseException],
              val: Optional[BaseException] = ...,
              tb: Optional[TracebackType] = ...):
        print(f'CorpusGenerator.throw: {typ}  {val}  {tb}')

    def close(self) -> None:
        print(f'CorpusGenerator.close')

    def send(self, value):
        print(f'CorpusGenerator.send: {value}')

    def __init__(self, x: List, y: List, batch_size: int = 64):
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self._index_list = list(range(len(self.x)))
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        sample_index = self._index_list[self._current_index * self.batch_size:
                                        (self._current_index + 1) * self.batch_size]
        self._current_index += 1
        if len(sample_index) == 0:
            raise StopIteration()
        return get_list_subset(self.x, sample_index), get_list_subset(self.y, sample_index)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


if __name__ == "__main__":
    pass
