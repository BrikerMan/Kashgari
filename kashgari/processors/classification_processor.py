from typing import List, Optional

import numpy as np
from tensorflow.python.keras.utils import to_categorical

import kashgari
from kashgari import utils
from kashgari.processors.base_processor import BaseProcessor


class ClassificationProcessor(BaseProcessor):
    """
    Corpus Pre Processor class
    """

    def __init__(self):
        super(ClassificationProcessor, self).__init__()

    def info(self):
        return {
            'task': kashgari.CLASSIFICATION
        }

    def _build_label_dict(self,
                          labels: List[str]):
        label_set = list(set(labels))

        self.label2idx = {self.token_pad: 0, }
        for idx, label in enumerate(label_set):
            self.label2idx[label] = idx + 1

        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)

    def process_y_dataset(self,
                          data: List[str],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if subset is not None:
            target = utils.get_list_subset(data, subset)
        else:
            target = data
        numerized_samples = self.numerize_label_sequences(target)
        return to_categorical(numerized_samples, len(self.label2idx))

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        result = []
        for seq in sequences:
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
        return result

    def numerize_label_sequences(self,
                                 sequences: List[str]) -> List[int]:
        """
        Convert label sequence to label-index sequence
        ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``

        Args:
            sequences: label sequence, list of str

        Returns:
            label-index sequence, list of int
        """
        return [self.label2idx[label] for label in sequences]

    def reverse_numerize_label_sequences(self, sequences, **kwargs):
        return [self.idx2label[label] for label in sequences]


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus

    x, y = SMP2018ECDTCorpus.load_data()
    p = ClassificationProcessor()
    p.analyze_corpus(x, y)
