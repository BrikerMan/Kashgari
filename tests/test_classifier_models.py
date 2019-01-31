# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: test_classifier_models.py
@time: 2019-01-27 13:28

"""
import time
import os
import random
import logging
import tempfile
import unittest

from kashgari.embeddings import WordEmbeddings, BERTEmbedding
from kashgari.tasks.classification import BLSTMModel, CNNModel, CNNLSTMModel, ClassificationModel
from kashgari.utils.logger import init_logger
init_logger()


SEQUENCE_LENGTH = 30

train_x = [
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]
train_y = ['a', 'a', 'a', 'b', 'c']

eval_x = [
    list('语言学是一门关于人类语言的科学研究。'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构研究与意义研究之间存在一个重要的主题划分。'),
    list('语法中包含了词法，句法以及语音。'),
    list('语音学是语言学的一个相关分支，它涉及到语音与非语音声音的实际属性，以及它们是如何发出与被接收到的。'),
    list('与学习语言不同，语言学是研究所有人类语文发展有关的一门学术科目。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

eval_y = ['a', 'a', 'a', 'b', 'c', 'a', 'c']


class EmbeddingManager(object):
    word2vec_embedding = None
    bert_embedding = None

    @classmethod
    def get_bert(cls):
        if cls.bert_embedding is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            bert_path = os.path.join(dir_path, 'data', 'test_bert_checkpoint')
            cls.bert_embedding = BERTEmbedding(bert_path, sequence_length=15)
            logging.info('bert_embedding seq len: {}'.format(cls.bert_embedding.sequence_length))
        return cls.bert_embedding

    @classmethod
    def get_w2v(cls):
        if cls.word2vec_embedding is None:
            cls.word2vec_embedding = WordEmbeddings('sgns.weibo.bigram', sequence_length=SEQUENCE_LENGTH, limit=5000)
        return cls.word2vec_embedding


class TestBLSTMModelModel(unittest.TestCase):
    model: ClassificationModel = None

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model = BLSTMModel()

    def test_build(self):
        self.model.fit(train_x, train_y, epochs=1)
        assert len(self.model.label2idx) == 4
        assert len(self.model.token2idx) > 4

    def test_fit(self):
        self.model.fit(train_x, train_y, eval_x, eval_y, epochs=self.epochs)

    def test_fit_class_weight(self):
        self.model.fit(train_x, train_y, eval_x, eval_y, class_weight=True, batch_size=128, epochs=2)

    def test_label_token_convert(self):
        self.test_fit()
        assert isinstance(self.model.convert_label_to_idx('a'), int)
        assert isinstance(self.model.convert_idx_to_label(1), str)
        assert all(isinstance(i, int) for i in self.model.convert_label_to_idx(['a']))
        assert all(isinstance(i, str) for i in self.model.convert_idx_to_label([1, 2]))

        sentence = random.choice(eval_x)
        tokens = self.model.embedding.tokenize(sentence)
        assert min(30, len(sentence)+2) == min(len(tokens), SEQUENCE_LENGTH)

    def test_predict(self):
        self.test_fit()
        sentence = list('语言学包含了几种分支领域。')
        assert isinstance(self.model.predict(sentence), str)
        assert isinstance(self.model.predict([sentence]), list)
        logging.info('test predict: {} -> {}'.format(sentence, self.model.predict(sentence)))

    def test_eval(self):
        self.test_fit()
        self.model.evaluate(eval_x, eval_y)

    def test_save_and_load(self):
        self.test_fit()
        model_path = os.path.join(tempfile.gettempdir(), 'kashgari_model', str(time.time()))
        self.model.save(model_path)
        new_model = BLSTMModel.load_model(model_path)
        assert new_model is not None
        sentence = list('语言学包含了几种分支领域。')
        result = new_model.predict(sentence)
        assert isinstance(result, str)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        logging.info('tearDownClass {}'.format(cls))


class TestBLSTMModelWithWord2Vec(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        embedding = EmbeddingManager.get_w2v()
        cls.model = BLSTMModel(embedding)


class TestBLSTMModelWithBERT(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        embedding = EmbeddingManager.get_bert()
        cls.model = BLSTMModel(embedding)

    def test_save_and_load(self):
        super(TestBLSTMModelWithBERT, self).test_save_and_load()


class TestCNNModel(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        TestCNNModel.model = CNNModel()

    def test_fit(self):
        super(TestCNNModel, self).test_fit()


class TestCNNModelWithWord2Vec(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        embedding = EmbeddingManager.get_w2v()
        cls.model = CNNModel(embedding)


class TestCNNModelWithBERT(TestBLSTMModelModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        embedding = EmbeddingManager.get_bert()
        TestCNNModelWithBERT.model = CNNModel(embedding)


class TestLSTMCNNModel(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model = CNNLSTMModel()


class TestLSTMCNNModelWithWord2Vec(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        embedding = EmbeddingManager.get_w2v()
        cls.model = CNNLSTMModel(embedding)


class TestLSTMCNNModelWithBERT(TestBLSTMModelModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        embedding = EmbeddingManager.get_bert()
        cls.model = CNNLSTMModel(embedding)


if __name__ == "__main__":
    unittest.main()
