# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: label_processor.py
# time: 2:53 下午

import collections
import operator

import numpy as np
import tqdm
from tensorflow.keras.utils import to_categorical
from typing import List

from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor


class ClassificationProcessor(ABCProcessor):

    def build_vocab_dict_if_needs(self, generator: CorpusGenerator):
        if not self.vocab2idx:
            vocab2idx = {}

            token2count = {}

            for _, label in tqdm.tqdm(generator, total=generator.steps, desc="Preparing classification label vocab dict"):
                count = token2count.get(label, 0)
                token2count[label] = count + 1

            sorted_token2count = sorted(token2count.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
            token2count = collections.OrderedDict(sorted_token2count)

            for token, token_count in token2count.items():
                if token not in vocab2idx:
                    vocab2idx[token] = len(vocab2idx)
            self.vocab2idx = vocab2idx
            self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])

    def numerize_samples(self,
                         samples: List[str],
                         seq_length: int = None,
                         one_hot: bool = False,
                         **kwargs) -> np.ndarray:
        sample_index = [self.vocab2idx[i] for i in samples]
        if one_hot:
            return to_categorical(sample_index, self.vocab_size)
        else:
            return np.array(sample_index)

    def reverse_numerize(self,
                         indexs: List[str],
                         lengths: List[int] = None,
                         **kwargs) -> List[str]:
        return [self.idx2vocab[i] for i in indexs]


if __name__ == "__main__":
    pass
