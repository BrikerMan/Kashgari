# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# version: 1.0
# license: Apache Licence
# file: corpus.py
# time: 2019-05-17 11:28

import operator
import collections
from typing import List, Union


class PreProcessor(object):
    """
    Corpus Pre Processor class
    """

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

        self.token2count = {}

        self.label2idx = {}
        self.idx2label = {}

    def build_token2idx_dict(self, tokenized_corpus: List[List[str]]):
        """
        Build token index dictionary using corpus

        Args:
            tokenized_corpus: List of tokenized sentences, like ``[['I', 'love', 'tf'], ...]``
        """
        token2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

        token2count = {}
        for sentence in tokenized_corpus:
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

    def build_classification_label2idx_dict(self,
                                            label_list: Union[List[str], List[List[str]]],
                                            multi_label: bool = False):
        """
        Build label2idx dict for classification labeling task

        Args:
            label_list: corpus label list
            multi_label: is multi-labeling task, default False.
        """
        pass

    def build_seq_labeling_label2idx_dict(self,
                                          label_list: List[List[str]]):
        """
        Build label2idx dict for sequence labeling task

        Args:
            label_list: corpus label list
        """
        pass


    def numerize_token_sequence(self,
                                sequence: List[str]) -> List[int]:
        """
        convert a token sequence to a numerical sequence, use the `self.token2idx` for mapping
        Args:
            sequence: a tokenized sequence

        Returns:
            Todo
        """
        pass

    def batch_numerize_token_sequence(self,
                                      sequence_list: List[List[str]]) -> List[List[int]]:
        """
        batch function of `numerize_token_sequence`
        Args:
            sequence_list: list of tokenized sequences

        Returns:
            Todo
        """
        return [self.numerize_token_sequence(seq) for seq in sequence_list]


    def numerize_label_sequence(self,
                                sequence: List[str]) -> List[int]:
        pass


    def preprocess_corpus(self, x, y, cache_dir):
        """
        Process data and dump to cached path
        Args:
            x:
            y:
            cache_dir:

        Returns:

        """


if __name__ == "__main__":
    print("Hello world")
