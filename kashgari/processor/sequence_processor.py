# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: text_processor.py
# time: 12:27 下午

import collections
import logging
import operator

import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from kashgari.generators import CorpusGenerator
from kashgari.processor.abc_processor import ABCProcessor
from kashgari.typing import TextSamplesVar, NumSamplesListVar


class SequenceProcessor(ABCProcessor):
    """
    Generic processor for the sequence samples.
    """

    def __init__(self, vocab_dict_type: str = 'text', **kwargs):
        """

        Args:
            vocab_dict_type: initial vocab dict type, one of `text` `labeling`.
            **kwargs:
        """
        super(SequenceProcessor, self).__init__(**kwargs)
        self.token_pad: str = kwargs.get('token_pad', '<PAD>')
        self.token_unk: str = kwargs.get('token_unk', '<UNK>')
        self.token_bos: str = kwargs.get('token_bos', '<BOS>')
        self.token_eos: str = kwargs.get('token_eos', '<EOS>')

        self.vocab2idx = {}
        self.idx2vocab = {}

        self.vocab_dict_type = vocab_dict_type

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

    def build_vocab_dict_if_needs(self, generator: CorpusGenerator, min_count: int = 3):
        if not self.vocab2idx:
            vocab2idx = self._initial_vocab_dic

            token2count = {}
            seq_lens = []
            generator.reset()
            for sentence, label in tqdm.tqdm(generator, total=generator.steps, desc="Preparing text vocab dict"):
                if self.vocab_dict_type == 'text':
                    target = sentence
                else:
                    target = label
                seq_lens.append(len(target))
                for token in target:
                    count = token2count.get(token, 0)
                    token2count[token] = count + 1

            sorted_token2count = sorted(token2count.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
            token2count = collections.OrderedDict(sorted_token2count)

            for token, token_count in token2count.items():
                if token not in vocab2idx and token_count >= min_count:
                    vocab2idx[token] = len(vocab2idx)
            self.vocab2idx = vocab2idx
            self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])

            if self.sequence_length is None:
                self.sequence_length = sorted(seq_lens)[int(0.95 * len(seq_lens))]
                logging.warning(f'Sequence length set to {self.sequence_length}')

            logging.info("------ Build vocab dict finished, Top 10 token ------")
            for token, index in list(self.vocab2idx.items())[:10]:
                logging.info(f"Token: {token:8s} -> {index}")
            logging.info("------ Build vocab dict finished, Top 10 token ------")
        else:
            if self.sequence_length is None:
                logging.debug('Start calculating the sequence length')
                seq_lens = []
                generator.reset()
                for sentence, _ in generator:
                    seq_lens.append(len(sentence))
                self.sequence_length = sorted(seq_lens)[int(0.95 * len(seq_lens))]
                logging.warning(f'Sequence length set to {self.sequence_length}')

    def numerize_samples(self, samples: TextSamplesVar, one_hot: bool = False) -> np.ndarray:
        if self.sequence_length is None:
            sequence_length = max([len(i) for i in samples])
            logging.warning(
                f'Sequence length is None, will use the max length of the samples, which is {sequence_length}')
        else:
            sequence_length = self.sequence_length

        numerized_samples = []
        for seq in samples:
            if self.vocab_dict_type == 'text':
                unk_index = self.vocab2idx[self.token_unk]
                numerized_samples.append([self.vocab2idx.get(token, unk_index) for token in seq])
            else:
                numerized_samples.append([self.vocab2idx[token] for token in seq])

        sample_index = pad_sequences(numerized_samples, sequence_length, padding='post', truncating='post')
        if one_hot:
            return to_categorical(sample_index, self.vocab_size)
        else:
            return np.array(sample_index)


if __name__ == "__main__":
    import logging
    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.generators import CorpusGenerator

    logging.basicConfig(level='DEBUG')
    x, y = ChineseDailyNerCorpus.load_data()
    gen = CorpusGenerator(x, y)
    p = SequenceProcessor(vocab_dict_type='labeling')
    p.build_vocab_dict_if_needs(gen)
    print(p.vocab2idx)

    p2 = SequenceProcessor()
    p2.build_vocab_dict_if_needs(gen)
    print(p2.vocab2idx)
