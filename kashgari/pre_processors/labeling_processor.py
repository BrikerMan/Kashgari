# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# version: 1.0
# license: Apache Licence
# file: corpus.py
# time: 2019-05-17 11:28

import collections
import logging
import operator
from typing import List, Dict, Tuple, Optional, Callable, Union

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

from kashgari import utils
from kashgari.pre_processors.base_processor import BaseProcessor


class LabelingProcessor(BaseProcessor):
    """
    Corpus Pre Processor class
    """

    def _build_label_dict(self,
                          label_list: List[List[str]]):
        """
        Build label2idx dict for sequence labeling task

        Args:
            label_list: corpus label list
        """
        label2idx: Dict[str: int] = {}

        token2count = {}
        for label_set in label_list:
            for sequence in label_set:
                for label in sequence:
                    count = token2count.get(label, 0)
                    token2count[label] = count + 1

        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token in token2count.keys():
            if token not in label2idx:
                label2idx[token] = len(label2idx)

        self.label2idx = label2idx
        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        logging.debug(f"build label2idx dict finished, contains {len(self.token2idx)} labels.")

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):

        result = []
        for seq in sequences:
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
        return result

    def numerize_label_sequences(self,
                                 sequences: List[List[str]]) -> List[List[int]]:
        result = []
        for seq in sequences:
            result.append([self.label2idx[label] for label in seq])
        return result

    def prepare_dicts_if_need(self,
                              corpus: List[List[str]],
                              labels: List[List[str]]):
        rec_seq_len = sorted([len(seq) for seq in corpus])[int(0.95 * len(corpus))]
        self.dataset_info['recommend_seq_len'] = rec_seq_len

        if not self.token2idx:
            self._build_token_dict(corpus)
        if not self.label2idx:
            self._build_label_dict(labels)

    def process_x_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        return self._process_sequence(self.numerize_token_sequences,
                                      data=data,
                                      maxlens=maxlens,
                                      subset=subset)

    def process_y_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        y = self._process_sequence(self.numerize_label_sequences,
                                   data=data,
                                   maxlens=maxlens,
                                   subset=subset)
        one_hot = to_categorical(y, len(self.label2idx))
        return one_hot

    def _process_sequence(self,
                          numerize_function: Callable,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Union[Tuple[np.ndarray, ...], List[np.ndarray]]:
        result = []
        for index, dataset in enumerate(data):
            if subset is not None:
                target = utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            numezied_target = numerize_function(target)
            target_maxlen = utils.get_tuple_item(maxlens, index)
            padded_target = pad_sequences(numezied_target, target_maxlen)
            result.append(padded_target)
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus

    x, y = ChineseDailyNerCorpus.load_data()
    p = LabelingProcessor()
    p.prepare_dicts_if_need(x, y)
    r = p.process_x_dataset((x,),
                            subset=[10, 12, 20])
    print(r)
