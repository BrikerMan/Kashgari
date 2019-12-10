# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_processor.py
# time: 2019-05-23 17:02
import os
import time
import logging
import tempfile
import unittest
import numpy as np
from kashgari import utils
from kashgari.processors import ClassificationProcessor, LabelingProcessor
from kashgari.corpus import SMP2018ECDTCorpus, ChineseDailyNerCorpus
from kashgari.tasks.classification import BiGRU_Model

ner_train_x, ner_train_y = ChineseDailyNerCorpus.load_data('valid')
class_train_x, class_train_y = SMP2018ECDTCorpus.load_data('valid')

sample_train_x = [
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

sample_train_y = [['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['c']]

sample_eval_x = [
    list('语言学是一门关于人类语言的科学研究。'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构研究与意义研究之间存在一个重要的主题划分。'),
    list('语法中包含了词法，句法以及语音。'),
    list('语音学是语言学的一个相关分支，它涉及到语音与非语音声音的实际属性，以及它们是如何发出与被接收到的。'),
    list('与学习语言不同，语言学是研究所有人类语文发展有关的一门学术科目。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

sample_eval_y = [['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['c'], ['b'], ['a']]


class TestLabelingProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = LabelingProcessor()
        cls.processor.analyze_corpus(ner_train_x, ner_train_y)

    def test_process_data(self):
        for i in range(3):
            vector_x = self.processor.process_x_dataset(ner_train_x[:20], max_len=12)
            assert vector_x.shape == (20, 12)

            subset_index = np.random.randint(0, len(ner_train_x), 30)
            vector_x = self.processor.process_x_dataset(ner_train_x, max_len=12, subset=subset_index)
            assert vector_x.shape == (30, 12)

            vector_y = self.processor.process_y_dataset(ner_train_y[:15], max_len=15)
            assert vector_y.shape == (15, 15, len(self.processor.label2idx))

            target_y = [seq[:15] for seq in ner_train_y[:15]]
            res_y = self.processor.reverse_numerize_label_sequences(vector_y.argmax(-1), lengths=np.full(15, 15))
            logging.info(f"target_y: {target_y}")
            logging.info(f"res_y: {res_y}")
            assert target_y == res_y

        self.processor.process_x_dataset(ner_train_x[:9], subset=[1, 2, 3])
        self.processor.process_y_dataset(ner_train_y[:9], subset=[1, 2, 3])


class TestClassificationProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = ClassificationProcessor()
        cls.processor.analyze_corpus(class_train_x, class_train_y)

    def test_process_data(self):
        for i in range(3):
            vector_x = self.processor.process_x_dataset(class_train_x[:20], max_len=12)
            assert vector_x.shape == (20, 12)

            subset_index = np.random.randint(0, len(class_train_x), 30)
            vector_x = self.processor.process_x_dataset(class_train_x, max_len=12, subset=subset_index)
            assert vector_x.shape == (30, 12)

            vector_y = self.processor.process_y_dataset(class_train_y[:15], max_len=15)
            assert vector_y.shape == (15, len(self.processor.label2idx))

            res_y = self.processor.reverse_numerize_label_sequences(vector_y.argmax(-1))
            assert class_train_y[:15] == res_y

    def test_multi_label_processor(self):
        p = ClassificationProcessor(multi_label=True)
        p.analyze_corpus(sample_train_x, sample_train_y)
        assert len(p.label2idx) == 3

        print(p.process_x_dataset(sample_train_x))
        print(p.process_y_dataset(sample_train_y))

    def test_load(self):
        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model = BiGRU_Model()
        model.fit(class_train_x, class_train_y, epochs=1)
        model.save(model_path)

        processor = utils.load_processor(model_path)

        assert processor.token2idx == model.embedding.processor.token2idx
        assert processor.label2idx == model.embedding.processor.label2idx

        assert processor.__class__ == model.embedding.processor.__class__

        process_x_0 = processor.process_x_dataset(class_train_x[:10])
        process_x_1 = model.embedding.process_x_dataset(class_train_x[:10])
        assert np.array_equal(process_x_0, process_x_1)


if __name__ == "__main__":
    print("Hello world")
