# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# version: 1.0
# license: Apache Licence
# file: corpus.py
# time: 2019-05-17 11:28

import collections
import json
import logging
import operator
import os
import pathlib
from typing import List, Dict


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

        self.token_pad = '<PAD>'
        self.token_unk = '<UNK>'
        self.token_bos = '<BOS>'
        self.token_eos = '<EOS>'
        self.seq_length_95 = None

    def _build_token2idx_dict(self, tokenized_corpus: List[List[str]]):
        """
        Build token index dictionary using corpus

        Args:
            tokenized_corpus: List of tokenized sentences, like ``[['I', 'love', 'tf'], ...]``
        """
        token2idx = {
            self.token_pad: 0,
            self.token_unk: 1,
            self.token_bos: 2,
            self.token_eos: 3
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
        logging.debug(f"build token2idx dict finished, contains {len(self.token2idx)} tokens.")


    # def _build_classification_label2idx_dict(self,
    #                                          label_list: Union[List[str], List[List[str]]],
    #                                          multi_label: bool = False):
    #     """
    #     Build label2idx dict for classification labeling task
    #
    #     Args:
    #         label_list: corpus label list
    #         multi_label: is multi-labeling task, default False.
    #     """
    #     pass

    def _build_seq_labeling_label2idx_dict(self,
                                           label_list: List[List[str]]):
        """
        Build label2idx dict for sequence labeling task

        Args:
            label_list: corpus label list
        """
        label2idx: Dict[str: int] = {}

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
        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        logging.debug(f"build label2idx dict finished, contains {len(self.token2idx)} labels.")

    def prepare_labeling_dicts_if_need(self,
                                       x: List[List[str]],
                                       y: List[List[str]]):
        """
        Process data and dump to cached path

        Args:
            x:
            y:

        Returns:

        """
        self.seq_length_95 = sorted([len(seq) for seq in x])[int(0.95 * len(x))]
        if not self.token2idx:
            self._build_token2idx_dict(x)
        if not self.label2idx:
            self._build_seq_labeling_label2idx_dict(y)

    def save_dicts(self, cache_dir: str):
        pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(cache_dir, 'token2idx.json'), 'w') as f:
            f.write(json.dumps(self.token2idx, ensure_ascii=False, indent=2))

        with open(os.path.join(cache_dir, 'label2idx.json'), 'w') as f:
            f.write(json.dumps(self.label2idx, ensure_ascii=False, indent=2))
        logging.debug(f"saved token2idx and label2idx to dir: {cache_dir}.")

    @classmethod
    def load_cached_processor(cls, cache_dir: str):
        processor = PreProcessor()
        with open(os.path.join(cache_dir, 'token2idx.json'), 'r') as f:
            processor.token2idx = json.loads(f.read())
            processor.idx2token = dict([(value, key) for key, value in processor.token2idx.items()])

        with open(os.path.join(cache_dir, 'label2idx.json'), 'r') as f:
            processor.label2idx = json.loads(f.read())
            processor.idx2label = dict([(value, key) for key, value in processor.label2idx.items()])
        logging.debug(f"loaded token2idx and label2idx from dir: {cache_dir}. "
                      f"Contain {len(processor.token2idx)} tokens and {len(processor.label2idx)} labels.")

        return processor

    def numerize_token_sequence(self,
                                sequence: List[str]) -> List[int]:
        """
        convert a token sequence to a numerical sequence, use the `self.token2idx` for mapping

        Args:
            sequence: a tokenized sequence

        Returns:
            numeric represent of the token sequence
        """
        idx_seq = []
        for token in sequence:
            index = self.token2idx.get(token, self.token2idx[self.token_unk])
            idx_seq.append(index)
        return idx_seq

    def numerize_label_sequence(self,
                                sequence: List[str]) -> List[int]:
        """
        Convert label sequence to label-index sequence
        ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``

        Args:
            sequence: label sequence, list of str

        Returns:
            label-index sequence, list of int
        """
        return [self.label2idx[label] for label in sequence]

    def reverse_numerize_label_sequence(self,
                                        sequence: List[int],
                                        length: int = 0) -> List[str]:
        """
        Convert label-index sequence to label sequence
        ``[0, 0, 2] -> ['O', 'O', 'B-ORG']``

        Args:
            sequence: label-index sequence
            length: max length of label sequence

        Returns:
            label sequence, list of str
        """
        token_seq = [self.idx2label[num] for num in sequence]
        if length != 0:
            token_seq = token_seq[:length]
        return token_seq


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus

    x, y = ChineseDailyNerCorpus.load_data()
    p = PreProcessor()
    p.prepare_labeling_dicts_if_need(x, y)
    p.save_dicts('./cache-dir')
    # p.build_token2idx_dict(x)
    # p.build_seq_labeling_label2idx_dict(y)
    # print(p.token2idx)
    # print(p.label2idx)
    # print(p.batch_numerize_token_sequence(x[:10]))
    # print("Hello world")
