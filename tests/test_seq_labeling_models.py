# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: test_seq_labeling_models.py
@time: 2019-01-27 13:55

"""
import os
import time
import logging
import tempfile
import unittest

from kashgari.embeddings import WordEmbeddings, BERTEmbedding, BaseEmbedding
from kashgari.tasks.seq_labeling import CNNLSTMModel, BLSTMModel, BLSTMCRFModel
from kashgari.utils.logger import init_logger

init_logger()


train_x = [
    ['我', '们', '变', '而', '以', '书', '会', '友', '，', '以', '书', '结', '缘', '，', '把', '欧', '美',
     '、', '港', '台', '流', '行', '的', '食', '品', '类', '图', '谱', '、', '画', '册', '、', '工', '具',
     '书', '汇', '集', '一', '堂', '。'],
    ['为', '了', '跟', '踪', '国', '际', '最', '新', '食', '品', '工', '艺', '、', '流', '行', '趋', '势',
     '，', '大', '量', '搜', '集', '海', '外', '专', '业', '书', '刊', '资', '料', '是', '提', '高', '技',
     '艺', '的', '捷', '径', '。'],
    ['其', '中', '线', '装', '古', '籍', '逾', '千', '册', '；', '民', '国', '出', '版', '物', '几', '百',
     '种', '；', '珍', '本', '四', '册', '、', '稀', '见', '本', '四', '百', '余', '册', '，', '出', '版',
     '时', '间', '跨', '越', '三', '百', '余', '年', '。']
]

train_y = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'B-LOC',
     'O', 'B-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O']
]

eval_x = train_x
eval_y = train_y

SEQUENCE_LENGTH = 15


class EmbeddingManager(object):
    word2vec_embedding = None
    bert_embedding = None

    @classmethod
    def get_bert(cls):
        if cls.bert_embedding is None:
            cls.bert_embedding = BERTEmbedding('chinese_L-12_H-768_A-12', sequence_length=SEQUENCE_LENGTH)
        return cls.bert_embedding

    @classmethod
    def get_w2v(cls):
        if cls.word2vec_embedding is None:
            cls.word2vec_embedding = WordEmbeddings('sgns.weibo.bigram', sequence_length=SEQUENCE_LENGTH, limit=5000)
        return cls.word2vec_embedding


class TestCNNLSTMModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model = CNNLSTMModel()

    def test_build(self):
        self.model.fit(train_x, train_y, epochs=1)
        self.assertEqual(len(self.model.label2idx), 5)
        self.assertGreater(len(self.model.token2idx), 4)

    def test_fit(self):
        self.model.fit(train_x, train_y, x_validate=eval_x, y_validate=eval_y, epochs=self.epochs)

    def test_label_token_convert(self):
        self.test_fit()
        sentence = list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分')
        idxs = self.model.embedding.tokenize(sentence)
        self.assertEqual(min(len(sentence), self.model.embedding.sequence_length),
                         min(len(idxs)-2, self.model.embedding.sequence_length))
        tokens = self.model.embedding.tokenize(sentence)
        self.assertEqual(len(sentence)+2, len(tokens))

    def test_predict(self):
        self.test_fit()
        sentence = list('语言学包含了几种分支领域。')
        result = self.model.predict(sentence)
        logging.info('test predict: {} -> {}'.format(sentence, result))
        self.assertTrue(isinstance(self.model.predict(sentence)[0], str))
        self.assertTrue(isinstance(self.model.predict([sentence])[0], list))
        self.assertEqual(len(self.model.predict(sentence)), len(sentence))

    def test_eval(self):
        self.test_fit()
        self.model.evaluate(train_x, train_y)

    def test_save_and_load(self):
        self.test_fit()
        model_path = os.path.join(tempfile.gettempdir(), 'kashgari_model', str(time.time()))
        self.model.save(model_path)
        new_model = BLSTMModel.load_model(model_path)
        self.assertIsNotNone(new_model)
        sentence = list('语言学包含了几种分支领域。')
        result = new_model.predict(sentence)
        self.assertTrue(isinstance(result[0], str))
        self.assertEqual(len(sentence), len(result))

    @classmethod
    def tearDownClass(cls):
        del cls.model
        logging.info('tearDownClass {}'.format(cls))


class TestCNNLSTMModelWithWord2Vec(TestCNNLSTMModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        embedding = EmbeddingManager.get_w2v()
        cls.model = CNNLSTMModel(embedding)


class TestLSTMCNNModelWithBERT(TestCNNLSTMModel):

    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        embedding = EmbeddingManager.get_bert()
        cls.model = CNNLSTMModel(embedding)


class TestBLSTMModel(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model = BLSTMModel()


class TestBLSTMModelWithWord2Vec(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        embedding = EmbeddingManager.get_w2v()
        cls.model = BLSTMModel(embedding)


class TestBLSTMModelWithBERT(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        embedding = EmbeddingManager.get_bert()
        cls.model = BLSTMModel(embedding)


class TestBLSTMCRFModel(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        cls.model = BLSTMCRFModel()


class TestBLSTMCRFModelWithWord2Vec(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 3
        embedding = EmbeddingManager.get_w2v()
        cls.model = BLSTMCRFModel(embedding)


class TestBLSTMCRFModelWithBERT(TestCNNLSTMModel):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        embedding = EmbeddingManager.get_bert()
        cls.model = BLSTMCRFModel(embedding)


if __name__ == "__main__":
    unittest.main()

