from typing import List, Tuple, Optional, Union

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

from kashgari import utils
from kashgari.pre_processors.base_processor import BaseProcessor


class ClassificationProcessor(BaseProcessor):
    """
    Corpus Pre Processor class
    """

    def __init__(self):
        super(ClassificationProcessor, self).__init__()

    def _build_label_dict(self,
                          labels: List[str]):
        label_set = []
        for item in labels:
            label_set += list(set(item))
        label_set = set(label_set)
        label2idx = {}
        for label in label_set:
            label2idx[label] = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)

    def process_x_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Union[Tuple[np.ndarray, ...], List[np.ndarray]]:
        result = []
        for index, dataset in enumerate(data):
            if subset is not None:
                target = utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            numerized_samples = self.numerize_token_sequences(target)
            target_maxlen = utils.get_tuple_item(maxlens, index)
            padded_target = pad_sequences(numerized_samples, target_maxlen)
            result.append(padded_target)
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)

    def process_y_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        result = []
        for index, dataset in enumerate(data):
            if subset is not None:
                target = utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            numerized_samples = self.numerize_label_sequences(target)
            one_hot_result = to_categorical(numerized_samples, len(self.label2idx))
            result.append(one_hot_result)
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)

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
    r = p.process_x_dataset((x, x), subset=[10, 12, 20], maxlens=(12, 20))
    print(r)
