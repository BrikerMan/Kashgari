# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_processor.py
# time: 2019-05-21 11:27

import os
import json
import logging
import pathlib
import operator
import collections
from typing import List, Tuple, Optional, Union
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from kashgari import utils
import numpy as np


class BaseProcessor(object):
    """
    Corpus Pre Processor class
    """

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

        self.token2count = {}

        self.label2idx = {}
        self.idx2label = {}

        self.token_pad = '<PAD>'
        self.token_unk = '<UNK>'
        self.token_bos = '<BOS>'
        self.token_eos = '<EOS>'

        self.dataset_info = {}

    def info(self):
        return {
            'task': ''
        }

    def analyze_corpus(self,
                       corpus: Union[List[List[str]]],
                       labels: Union[List[List[str]], List[str]],
                       force: bool = False):
        rec_len = sorted([len(seq) for seq in corpus])[int(0.95 * len(corpus))]
        self.dataset_info['RECOMMEND_LEN'] = rec_len

        if not self.token2idx or force:
            self._build_token_dict(corpus)
        if not self.label2idx or force:
            self._build_label_dict(labels)

    def save_dicts(self, cache_dir: str):
        pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(cache_dir, 'token2idx.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.token2idx, ensure_ascii=False, indent=2))

        with open(os.path.join(cache_dir, 'label2idx.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.label2idx, ensure_ascii=False, indent=2))
        logging.debug(f"saved token2idx and label2idx to dir: {cache_dir}.")

    @classmethod
    def load_cached_processor(cls, cache_dir: str):
        processor = cls()
        with open(os.path.join(cache_dir, 'token2idx.json'), 'r', encoding='utf-8') as f:
            processor.token2idx = json.loads(f.read())
            processor.idx2token = dict([(value, key) for key, value in processor.token2idx.items()])

        with open(os.path.join(cache_dir, 'label2idx.json'), 'r', encoding='utf-8') as f:
            processor.label2idx = json.loads(f.read())
            processor.idx2label = dict([(value, key) for key, value in processor.label2idx.items()])
        logging.debug(f"loaded token2idx and label2idx from dir: {cache_dir}. "
                      f"Contain {len(processor.token2idx)} tokens and {len(processor.label2idx)} labels.")

        return processor

    def _build_token_dict(self, corpus: List[List[str]]):
        """
        Build token index dictionary using corpus

        Args:
            corpus: List of tokenized sentences, like ``[['I', 'love', 'tf'], ...]``
        """
        token2idx = {
            self.token_pad: 0,
            self.token_unk: 1,
            self.token_bos: 2,
            self.token_eos: 3
        }

        token2count = {}
        for sentence in corpus:
            for token in sentence:
                count = token2count.get(token, 0)
                token2count[token] = count + 1

        # 按照词频降序排序
        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token in token2count.keys():
            if token not in token2idx:
                token2idx[token] = len(token2idx)

        self.token2idx = token2idx
        self.idx2token = dict([(value, key) for key, value in self.token2idx.items()])
        logging.debug(f"build token2idx dict finished, contains {len(self.token2idx)} tokens.")
        self.dataset_info['token_count'] = len(self.token2idx)

    def _build_label_dict(self, corpus: Tuple):
        raise NotImplementedError

    def process_x_dataset(self,
                          data: List[List[str]],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if subset is not None:
            target = utils.get_list_subset(data, subset)
        else:
            target = data
        numerized_samples = self.numerize_token_sequences(target)

        return pad_sequences(numerized_samples, max_len, padding='post', truncating='post')

    def process_y_dataset(self,
                          data: Union[List[List[str]], List[str]],
                          max_len: Optional[int],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        raise NotImplementedError

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        raise NotImplementedError

    def numerize_label_sequences(self,
                                 sequences: List[List[str]]) -> List[List[int]]:
        raise NotImplementedError

    def reverse_numerize_label_sequences(self, sequence, **kwargs):
        raise NotImplemented

    def __repr__(self):
        return f"<{self.__class__}>"

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    print("Hello world")
