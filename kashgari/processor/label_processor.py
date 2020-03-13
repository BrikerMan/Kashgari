# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: label_processor.py
# time: 2:53 下午

import operator
import collections
from typing import Generator
from kashgari.processor.abs_processor import ABCProcessor


class TextProcessor(ABCProcessor):
    def __init__(self, **kwargs):
        super(ABCProcessor, self).__init__(**kwargs)
        self.vocab2idx = {}
        self.idx2vocab = {}

    def build_vocab_dict(self, generator: Generator, min_count: int=3):
        generator.reset()
        if not self.vocab2idx:
            vocab2idx = {}

            token2count = {}

            for _, y_set in generator:
                for label in y_set:
                    count = token2count.get(label, 0)
                    token2count[label] = count + 1

            sorted_token2count = sorted(token2count.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
            token2count = collections.OrderedDict(sorted_token2count)

            for token, token_count in token2count.items():
                if token not in vocab2idx and token_count >= min_count:
                    vocab2idx[token] = len(vocab2idx)
            self.vocab2idx = vocab2idx
            self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])


if __name__ == "__main__":
    pass
