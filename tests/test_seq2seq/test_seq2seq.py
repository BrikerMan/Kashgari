# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_seq2seq.py
# time: 4:46 下午

import unittest
from kashgari.tasks.seq2seq import Seq2Seq
from kashgari.corpus import ChineseDailyNerCorpus


class TestSeq2Seq(unittest.TestCase):
    def test_base_use_case(self):
        x, y = ChineseDailyNerCorpus.load_data('test')
        x = x[:200]
        y = y[:200]
        seq2seq = Seq2Seq()
        seq2seq.fit(x, y)
        print(seq2seq.predict(x))
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
