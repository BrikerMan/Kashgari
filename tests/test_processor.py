# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_processor.py
# time: 2019-05-23 17:02

import unittest
import numpy as np
from kashgari.processors import ClassificationProcessor, LabelingProcessor
from kashgari.corpus import SMP2018ECDTCorpus, ChineseDailyNerCorpus

train_x, train_y = ChineseDailyNerCorpus.load_data('valid')
train_x1, train_y1 = SMP2018ECDTCorpus.load_data('valid')


class TestLabelingProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = LabelingProcessor()
        cls.processor.analyze_corpus(train_x, train_y)

    def test_process_data(self):
        for i in range(3):
            vector_x = self.processor.process_x_dataset(train_x[:20], max_len=12)
            assert vector_x.shape == (20, 12)

            subset_index = np.random.randint(0, len(train_x), 30)
            vector_x = self.processor.process_x_dataset(train_x, max_len=12, subset=subset_index)
            assert vector_x.shape == (30, 12)

            vector_y = self.processor.process_y_dataset(train_y[:15], max_len=15)
            assert vector_y.shape == (15, 15, len(self.processor.label2idx))

            target_y = [seq[:15] for seq in train_y[:15]]
            res_y = self.processor.reverse_numerize_label_sequences(vector_y.argmax(-1), lengths=np.full(15, 15))
            assert target_y == res_y

        self.processor.process_x_dataset(train_x[:9], subset=[1, 2, 3])
        self.processor.process_y_dataset(train_y[:9], subset=[1, 2, 3])


class TestClassificationProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = ClassificationProcessor()
        cls.processor.analyze_corpus(train_x1, train_y1)

    def test_process_data(self):
        for i in range(3):
            vector_x = self.processor.process_x_dataset(train_x1[:20], max_len=12)
            assert vector_x.shape == (20, 12)

            subset_index = np.random.randint(0, len(train_x1), 30)
            vector_x = self.processor.process_x_dataset(train_x1, max_len=12, subset=subset_index)
            assert vector_x.shape == (30, 12)

            vector_y = self.processor.process_y_dataset(train_y1[:15], max_len=15)
            assert vector_y.shape == (15, len(self.processor.label2idx))

            res_y = self.processor.reverse_numerize_label_sequences(vector_y.argmax(-1))
            assert train_y1[:15] == res_y


if __name__ == "__main__":
    print("Hello world")
