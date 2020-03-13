# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_processor.py
# time: 2:53 下午

from abc import ABC
from typing import Generator


class ABCProcessor(ABC):
    def __init__(self, **kwargs):
        self.vocab2idx = {}
        self.idx2vocab = {}

        self.sequence_length = None

        print('ABCProcessorABCProcessorABCProcessorABCProcessorABCProcessorABCProcessor')

    @property
    def vocab_size(self) -> int:
        return len(self.vocab2idx)

    @property
    def is_vocab_build(self) -> bool:
        return self.vocab_size != 0

    def build_vocab_dict_if_needs(self, generator: Generator, min_count: int = 3):
        raise NotImplementedError


if __name__ == "__main__":
    pass
