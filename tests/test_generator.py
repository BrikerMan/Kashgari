# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_generator.py
# time: 5:46 下午

import unittest
from tests.test_macros import TestMacros

from kashgari.generators import CorpusGenerator


class TestGenerator(unittest.TestCase):
    def test_corpus_generator(self):
        x_set, y_set = TestMacros.load_labeling_corpus()
        corpus_gen = CorpusGenerator(x_set, y_set)

        for x, y in corpus_gen:
            print(x, y)

    def test_batch_generator(self):
        x_set, y_set = [], []
        for i in range(100):
            x_set.append([f'x{i}'] * 4)
            y_set.append([f'y{i}'] * 4)
        corpus_gen = CorpusGenerator(x_set, y_set)

        a = []
        for x, y in corpus_gen:
            print(x, y)
            a.append(x[0])

        print(sorted(a))


if __name__ == '__main__':
    unittest.main()
