# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_processor.py
# time: 2:53 下午

from abc import ABC
from typing import Generator, Dict


class ABCProcessor(ABC):
    def info(self) -> Dict:
        return {
            'config': {
                'vocab2idx': self.vocab2idx,
            },
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__
        }

    def __init__(self, **kwargs):
        self.vocab2idx = kwargs.get('vocab2idx', {})
        self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab2idx)

    @property
    def is_vocab_build(self) -> bool:
        return self.vocab_size != 0

    def build_vocab_dict_if_needs(self, generator: Generator):
        raise NotImplementedError


if __name__ == "__main__":
    pass
