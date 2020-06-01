# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_sequence_processor.py
# time: 12:09 下午

import random
import unittest
from tests.test_macros import TestMacros

from kashgari.utils import load_data_object
from kashgari.processors import SequenceProcessor


class TestSequenceProcessor(unittest.TestCase):
    def test_text_processor(self):
        x_set, y_set = TestMacros.load_labeling_corpus()
        x_samples = random.sample(x_set, 5)
        text_processor = SequenceProcessor(min_count=1)
        text_processor.build_vocab(x_set, y_set)
        text_idx = text_processor.transform(x_samples)

        text_info_dict = text_processor.to_dict()
        text_processor2: SequenceProcessor = load_data_object(text_info_dict)

        text_idx2 = text_processor2.transform(x_samples)
        sample_lengths = [len(i) for i in x_samples]

        assert (text_idx2 == text_idx).all()
        assert text_processor.inverse_transform(text_idx, lengths=sample_lengths) == x_samples
        assert text_processor2.inverse_transform(text_idx2, lengths=sample_lengths) == x_samples

    def test_label_processor(self):
        x_set, y_set = TestMacros.load_labeling_corpus()
        text_processor = SequenceProcessor(build_vocab_from_labels=True, min_count=1)
        text_processor.build_vocab(x_set, y_set)

        samples = random.sample(y_set, 20)

        text_idx = text_processor.transform(samples)

        text_info_dict = text_processor.to_dict()

        text_processor2: SequenceProcessor = load_data_object(text_info_dict)

        text_idx2 = text_processor2.transform(samples)
        lengths = [len(i) for i in samples]
        assert (text_idx2 == text_idx).all()
        assert text_processor2.inverse_transform(text_idx, lengths=lengths) == samples
        assert text_processor2.inverse_transform(text_idx2, lengths=lengths) == samples

        text_idx3 = text_processor.transform(samples, seq_length=20)
        assert [len(i) for i in text_idx3] == [20] * len(text_idx3)


if __name__ == "__main__":
    pass
