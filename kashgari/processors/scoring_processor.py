# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: scoring_processor.py
# time: 11:10 上午

from typing import List, Optional

import numpy as np

import kashgari
from kashgari import utils
from kashgari.processors.base_processor import BaseProcessor


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


class ScoringProcessor(BaseProcessor):
    """
    Corpus Pre Processor class
    """

    def __init__(self, output_dim=None, **kwargs):
        super(ScoringProcessor, self).__init__(**kwargs)
        self.output_dim = output_dim

    def info(self):
        info = super(ScoringProcessor, self).info()
        info['task'] = kashgari.SCORING
        return info

    def _build_label_dict(self,
                          label_list: List[List[float]]):
        """
        Build label2idx dict for sequence labeling task

        Args:
            label_list: corpus label list
        """
        if self.output_dim is None:
            label_sample = label_list[0]
            if isinstance(label_sample, np.ndarray) and len(label_sample.shape) == 1:
                self.output_dim = label_sample.shape[0]
            elif is_numeric(label_sample):
                self.output_dim = 1
            elif isinstance(label_sample, list):
                self.output_dim = len(label_sample)
            else:
                raise ValueError('Scoring Label Sample must be a float, float array or 1D numpy array')
        # np_labels = np.array(label_list)
        # if np_labels.max() > 1 or np_labels.min() < 0:
        #     raise ValueError('Scoring Label Sample must be in range[0,1]')

    def process_y_dataset(self,
                          data: List[List[str]],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if subset is not None:
            target = utils.get_list_subset(data, subset)
        else:
            target = data[:]
        y = np.array(target)
        return y

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
        return sequences

    def reverse_numerize_label_sequences(self,
                                         sequences,
                                         lengths=None):
        return sequences


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus

    x, y = SMP2018ECDTCorpus.load_data()
    x = x[:3]
    y = [0.2, 0.3, 0.2]
    p = ScoringProcessor()
    p.analyze_corpus(x, y)
    print(p.process_y_dataset(y))
