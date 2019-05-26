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
from typing import List, Dict, Optional

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

import kashgari
from kashgari import utils
from kashgari.processors.base_processor import BaseProcessor


class LabelingProcessor(BaseProcessor):
    """
    Corpus Pre Processor class
    """

    def info(self):
        info = super(LabelingProcessor, self).info()
        info['task'] = kashgari.LABELING
        return info

    def _build_label_dict(self,
                          label_list: List[List[str]]):
        """
        Build label2idx dict for sequence labeling task

        Args:
            label_list: corpus label list
        """
        label2idx: Dict[str: int] = {
            self.token_pad: 0
        }

        token2count = {}

        for sequence in label_list:
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
        self.idx2label = dict([(value, key)
                               for key, value in self.label2idx.items()])
        logging.debug(f"build label2idx dict finished, contains {len(self.label2idx)} labels.")

    def process_y_dataset(self,
                          data: List[List[str]],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if subset is not None:
            target = utils.get_list_subset(data, subset)
        else:
            target = data[:]
        numerized_samples = self.numerize_label_sequences(target)
        padded_seq = pad_sequences(
            numerized_samples, max_len, padding='post', truncating='post')
        return to_categorical(padded_seq, len(self.label2idx))

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):

        result = []
        for seq in sequences:
            if self.add_bos_eos:
                seq = [self.token_bos] + seq + [self.token_eos]
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
        return result

    def numerize_label_sequences(self,
                                 sequences: List[List[str]]) -> List[List[int]]:
        result = []
        for seq in sequences:
            if self.add_bos_eos:
                seq = [self.token_pad] + seq + [self.token_pad]
            result.append([self.label2idx[label] for label in seq])
        return result

    def reverse_numerize_label_sequences(self,
                                         sequences,
                                         lengths=None):
        result = []

        for index, seq in enumerate(sequences):
            labels = []
            if self.add_bos_eos:
                seq = seq[1:]
            for idx in seq:
                labels.append(self.idx2label[idx])
            if lengths is not None:
                labels = labels[:lengths[index]]
            result.append(labels)
        return result


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus

    x, y = ChineseDailyNerCorpus.load_data()
    p = LabelingProcessor()
    p.analyze_corpus(x, y)
    r = p.process_x_dataset(x, subset=[10, 12, 20])
    print(r)
