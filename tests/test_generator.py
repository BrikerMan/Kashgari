# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_generator.py
# time: 5:46 下午

import unittest
from tests.test_macros import TestMacros

from kashgari.generators import CorpusGenerator, BatchDataGenerator


class TestGenerator(unittest.TestCase):
    def test_corpus_generator(self):
        x_set, y_set = TestMacros.load_labeling_corpus('custom_1')
        corpus_gen = CorpusGenerator(x_set, y_set)

        for x, y in corpus_gen:
            print(x, y)

    def test_batch_generator(self):
        from kashgari.processors import SequenceProcessor
        x_set, y_set = [], []
        for i in range(22):
            x_set.append([f'x{i}'] * 4)
            y_set.append([f'y{i}'] * 4)
        corpus_gen = CorpusGenerator(x_set, y_set, shuffle=False)

        a = []
        for x, y in corpus_gen:
            print(x, y)
            a.append(x[0])

        print(sorted(a))

        p1 = SequenceProcessor(min_count=1)
        p1.build_vocab_dict_if_needs(corpus_gen)
        p2 = SequenceProcessor(vocab_dict_type='labeling', min_count=1)
        p2.build_vocab_dict_if_needs(corpus_gen)

        batch_gen = BatchDataGenerator(corpus_gen,
                                       text_processor=p1,
                                       label_processor=p2,
                                       seq_length=5,
                                       batch_size=4)
        print('------ Iterator --------')
        for i in batch_gen:
            x, y = i
            print(x)

        print('------ Generator --------')
        gen = batch_gen.generator()
        try:
            while True:
                x, y = next(gen)
                print(x)
        except StopIteration:
            pass

if __name__ == '__main__':
    unittest.main()
