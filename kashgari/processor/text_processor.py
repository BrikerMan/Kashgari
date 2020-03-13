# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: text_processor.py
# time: 12:27 下午

import operator
import collections
import logging
from kashgari.processor.abs_processor import ABCProcessor
from kashgari.typing import TextSamplesVar, NumSamplesListVar
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kashgari.generator import CorpusGenerator


class TextProcessor(ABCProcessor):
    def __init__(self, **kwargs):
        super(TextProcessor, self).__init__(**kwargs)
        self.token_pad: str = kwargs.get('token_pad', '<PAD>')
        self.token_unk: str = kwargs.get('token_unk', '<UNK>')
        self.token_bos: str = kwargs.get('token_bos', '<BOS>')
        self.token_eos: str = kwargs.get('token_eos', '<EOS>')

        self.vocab2idx = {}
        self.idx2vocab = {}

    def build_vocab_dict_if_needs(self, generator: CorpusGenerator, min_count: int = 3):
        if not self.vocab2idx:
            vocab2idx = {
                self.token_pad: 0,
                self.token_unk: 1,
                self.token_bos: 2,
                self.token_eos: 3
            }

            token2count = {}
            seq_lens = []
            generator.reset()
            for sentence, _ in generator:
                seq_lens.append(len(sentence))
                for token in sentence:
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
        else:
            if self.sequence_length is None:
                logging.debug('Start calculating the sequence length')
                seq_lens = []
                generator.reset()
                for sentence, _ in generator:
                    seq_lens.append(len(sentence))
                self.sequence_length = sorted(seq_lens)[int(0.95 * len(seq_lens))]
                logging.debug(f'Sequence length set to {self.sequence_length}')

    def numerize_samples(self, samples: TextSamplesVar) -> NumSamplesListVar:
        if self.sequence_length is None:
            sequence_length = max([len(i) for i in samples])
        else:
            sequence_length = self.sequence_length

        numerized_samples = []
        for seq in samples:
            unk_index = self.vocab2idx[self.token_unk]
            numerized_samples.append([self.vocab2idx.get(token, unk_index) for token in seq])

        return pad_sequences(numerized_samples, sequence_length, padding='post', truncating='post')


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.utils import CorpusGenerator

    x, y = ChineseDailyNerCorpus.load_data()
    gen = CorpusGenerator(x, y)
    p = TextProcessor()
    p.build_vocab_dict_if_needs(gen)
    print(p.vocab2idx)
