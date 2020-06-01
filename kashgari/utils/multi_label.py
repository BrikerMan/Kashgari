# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: multi_label.py
# time: 11:23 上午


from typing import List, Dict

import numpy as np

from kashgari.types import MultiLabelClassificationLabelVar


class MultiLabelBinarizer:
    def __init__(self, vocab2idx: Dict[str, int]):
        self.vocab2idx = vocab2idx
        self.idx2vocab = dict([(v, k) for k, v in vocab2idx.items()])

    @property
    def classes(self) -> List[str]:
        return list(self.idx2vocab.values())

    def transform(self, samples: MultiLabelClassificationLabelVar) -> np.ndarray:
        data = np.zeros((len(samples), len(self.vocab2idx)))
        for sample_index, sample in enumerate(samples):
            for label in sample:
                data[sample_index][self.vocab2idx[label]] = 1
        return data

    def inverse_transform(self, preds: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        data = []
        for sample in preds:
            x = []
            for label_x in np.where(sample >= threshold)[0]:
                x.append(self.idx2vocab[label_x])
            data.append(x)
        return data


if __name__ == "__main__":
    pass
