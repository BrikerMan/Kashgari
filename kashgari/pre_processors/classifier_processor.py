import collections
import json
import logging
import operator
import os
import pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from kashgari import utils


class ClassifierProcessor(object):
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

    def _build_label_dict(self,
                          labels: List[str]):
        label_set = set(labels)
        label2idx = {}
        for label in label_set:
            label2idx[label] = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)

    def prepare_dicts_if_need(self,
                              corpus: List[List[str]],
                              labels: List[str]):
        rec_seq_len = sorted([len(seq) for seq in corpus])[int(0.95 * len(corpus))]
        self.dataset_info['recommend_seq_len'] = rec_seq_len

        if not self.token2idx:
            self._build_token_dict(corpus)
        if not self.label2idx:
            self._build_label_dict(labels)

    def save_dicts(self, cache_dir: str):
        pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(cache_dir, 'token2idx.json'), 'w') as f:
            f.write(json.dumps(self.token2idx, ensure_ascii=False, indent=2))

        with open(os.path.join(cache_dir, 'label2idx.json'), 'w') as f:
            f.write(json.dumps(self.label2idx, ensure_ascii=False, indent=2))
        logging.debug(f"saved token2idx and label2idx to dir: {cache_dir}.")

    @classmethod
    def load_cached_processor(cls, cache_dir: str):
        processor = cls()
        with open(os.path.join(cache_dir, 'token2idx.json'), 'r') as f:
            processor.token2idx = json.loads(f.read())
            processor.idx2token = dict([(value, key) for key, value in processor.token2idx.items()])

        with open(os.path.join(cache_dir, 'label2idx.json'), 'r') as f:
            processor.label2idx = json.loads(f.read())
            processor.idx2label = dict([(value, key) for key, value in processor.label2idx.items()])
        logging.debug(f"loaded token2idx and label2idx from dir: {cache_dir}. "
                      f"Contain {len(processor.token2idx)} tokens and {len(processor.label2idx)} labels.")

        return processor

    def process_x_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        result = []
        for index, dataset in enumerate(data):
            if subset:
                target = utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            numezied_target = self.numerize_token_sequences(target)
            target_maxlen = utils.get_tuple_item(maxlens, index)
            padded_target = pad_sequences(numezied_target, target_maxlen)
            result.append(padded_target)
        return tuple(result)

    def process_y_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        result = []
        for index, dataset in enumerate(data):
            if subset:
                target = utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            numezied_target = self.numerize_token_sequences(target)
            one_hot_result = to_categorical(numezied_target, len(self.label2idx))
            result.append(one_hot_result)
        return tuple(result)

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        result = []
        for seq in sequences:
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
        return result

    def numerize_label_sequences(self,
                                sequences: List[List[str]]) -> List[List[int]]:
        """
        Convert label sequence to label-index sequence
        ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``

        Args:
            sequence: label sequence, list of str

        Returns:
            label-index sequence, list of int
        """
        result = []
        for sequence in sequences:
            result.append([self.label2idx[label] for label in sequence])
        return result


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus

    x, y = SMP2018ECDTCorpus.load_data()
    p = ClassifierProcessor()
    p.prepare_dicts_if_need(x, y)
    r = p.process_x_dataset((x,), subset=[10, 12, 20], maxlens=(12,))
    print(r)
