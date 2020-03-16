# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: text_processor.py
# time: 12:27 下午

import logging
import collections
import operator

import tqdm
import numpy as np
from typing import Dict, List

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor
from kashgari.types import TextSamplesVar


class SequenceProcessor(ABCProcessor):
    """
    Generic processors for the sequence samples.
    """

    def info(self) -> Dict:
        data = super(SequenceProcessor, self).info()
        data['config'].update({
            'vocab2idx': self.vocab2idx,
            'token_pad': self.token_pad,
            'token_unk': self.token_unk,
            'token_bos': self.token_bos,
            'token_eos': self.token_eos,
            'vocab_dict_type': self.vocab_dict_type,
            'min_count': self.min_count
        })
        return data

    def __init__(self,
                 vocab_dict_type: str = 'text',
                 min_count: int = 3,
                 **kwargs):
        """

        Args:
            vocab_dict_type: initial vocab dict type, one of `text` `labeling`.
            **kwargs:
        """
        super(SequenceProcessor, self).__init__(**kwargs)
        self.token_pad: str = kwargs.get('token_pad', '[PAD]')
        self.token_unk: str = kwargs.get('token_unk', '[UNK]')
        self.token_bos: str = kwargs.get('token_bos', '[BOS]')
        self.token_eos: str = kwargs.get('token_eos', '[EOS]')

        self.vocab_dict_type = vocab_dict_type
        self.min_count = min_count

        if vocab_dict_type == 'text':
            self._initial_vocab_dic = {
                self.token_pad: 0,
                self.token_unk: 1,
                self.token_bos: 2,
                self.token_eos: 3
            }
        elif vocab_dict_type == 'labeling':
            self._initial_vocab_dic = {
                self.token_pad: 0
            }
        else:
            self._initial_vocab_dic = {}

        self._showed_seq_len_warning = False

    def build_vocab_dict_if_needs(self, generator: CorpusGenerator):
        if not self.vocab2idx:
            vocab2idx = self._initial_vocab_dic

            token2count = {}

            for sentence, label in tqdm.tqdm(generator, total=generator.steps, desc="Preparing text vocab dict"):
                if self.vocab_dict_type == 'text':
                    target = sentence
                else:
                    target = label
                for token in target:
                    count = token2count.get(token, 0)
                    token2count[token] = count + 1

            sorted_token2count = sorted(token2count.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
            token2count = collections.OrderedDict(sorted_token2count)

            for token, token_count in token2count.items():
                if token not in vocab2idx and token_count >= self.min_count:
                    vocab2idx[token] = len(vocab2idx)
            self.vocab2idx = vocab2idx
            self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])

            logging.info("------ Build vocab dict finished, Top 10 token ------")
            for token, index in list(self.vocab2idx.items())[:10]:
                logging.info(f"Token: {token:8s} -> {index}")
            logging.info("------ Build vocab dict finished, Top 10 token ------")

    def numerize_samples(self,
                         samples: TextSamplesVar,
                         seq_length: int = None,
                         segment: bool = False,
                         one_hot: bool = False,
                         **kwargs) -> np.ndarray:
        if seq_length is None:
            seq_length = max([len(i) for i in samples])
            if not self._showed_seq_len_warning:
                logging.warning(
                    f'Sequence length is None, will use the max length of the samples, which is {seq_length}')
                self._showed_seq_len_warning = True

        numerized_samples = []
        for seq in samples:
            if self.vocab_dict_type == 'text':
                unk_index = self.vocab2idx[self.token_unk]
                numerized_samples.append([self.vocab2idx.get(token, unk_index) for token in seq])
            else:
                numerized_samples.append([self.vocab2idx[token] for token in seq])

        sample_index = pad_sequences(numerized_samples, seq_length, padding='post', truncating='post')
        if one_hot:
            token_ids = to_categorical(sample_index, self.vocab_size)
        else:
            token_ids = np.array(sample_index)

        if segment:
            segment_ids = np.zeros(token_ids.shape, dtype=np.int32)
            return token_ids, segment_ids
        else:
            return token_ids

    def reverse_numerize(self,
                         indexs: List[str],
                         lengths: List[int] = None,
                         **kwargs) -> List[List[str]]:
        result = []
        for index, seq in enumerate(indexs):
            labels = []
            for idx in seq:
                labels.append(self.idx2vocab[idx])
            if lengths is not None:
                labels = labels[:lengths[index]]
            result.append(labels)
        return result


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.generators import CorpusGenerator

    logging.basicConfig(level='DEBUG')
    x, y = ChineseDailyNerCorpus.load_data()
    gen = CorpusGenerator(x, y)
    p = SequenceProcessor(vocab_dict_type='labeling')
    p.build_vocab_dict_if_needs(gen)
    print(p.vocab2idx)

