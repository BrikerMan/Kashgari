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

    def build_vocab_dict(self, generator: Generator, min_count: int=3):
        raise NotImplementedError


if __name__ == "__main__":
    pass
