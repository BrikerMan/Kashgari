# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: embedding_test
@time: 2019-01-22

"""

import logging
import unittest

from kashgari import k
from kashgari.embeddings import WordEmbeddings, BERTEmbedding, CustomEmbedding

SEQUENCE_LENGTH = 30


class WordEmbeddingsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(WordEmbeddingsTest, self).__init__(*args, **kwargs)
        self.embedding = WordEmbeddings('sgns.weibo.bigram',
                                        sequence_length=SEQUENCE_LENGTH,
                                        limit=1000)

    def test_build(self):
        self.assertEqual(self.embedding.token_count, 1004)
        self.assertEqual(self.embedding.idx2token[0], k.PAD)
        self.assertEqual(self.embedding.idx2token[1], k.BOS)
        self.assertEqual(self.embedding.idx2token[2], k.EOS)
        self.assertEqual(self.embedding.idx2token[3], k.UNK)

    def test_tokenize(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        tokens = self.embedding.tokenize(sentence)

        logging.info('tokenize test: {} -> {}'.format(sentence, tokens))
        self.assertEqual(len(tokens), len(tokens))
        self.assertEqual(tokens[-1], 3, msg='check unk value')

        token_list = self.embedding.tokenize([sentence])
        self.assertEqual(len(token_list[0]), len(sentence))

    def test_embed(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed(sentence)
        logging.info('embed test: {} -> {}'.format(sentence, embedded_sentence))
        self.assertEqual(embedded_sentence.shape, (SEQUENCE_LENGTH, self.embedding.embedding_size))

        embedded_sentences = self.embedding.embed([sentence])
        self.assertEqual(embedded_sentences.shape, (1, SEQUENCE_LENGTH, self.embedding.embedding_size))


class BertEmbeddingsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BertEmbeddingsTest, self).__init__(*args, **kwargs)
        self.embedding = BERTEmbedding('chinese_L-12_H-768_A-12',
                                       sequence_length=SEQUENCE_LENGTH)

    def test_build(self):
        self.assertGreater(self.embedding.embedding_size, 0)
        self.assertEqual(self.embedding.token2idx[k.PAD], 0)
        self.assertGreater(self.embedding.token2idx[k.BOS], 0)
        self.assertGreater(self.embedding.token2idx[k.EOS], 0)
        self.assertGreater(self.embedding.token2idx[k.UNK], 0)

    def test_tokenize(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        tokens = self.embedding.tokenize(sentence)

        logging.info('tokenize test: {} -> {}'.format(sentence, tokens))
        self.assertEqual(len(tokens), len(tokens))

        token_list = self.embedding.tokenize([sentence])
        self.assertEqual(len(token_list[0]), len(sentence))

    def test_embed(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed(sentence)
        logging.info('embed test: {} -> {}'.format(sentence, embedded_sentence))
        self.assertEqual(embedded_sentence.shape, (SEQUENCE_LENGTH, self.embedding.embedding_size))

        embedded_sentences = self.embedding.embed([sentence])
        self.assertEqual(embedded_sentences.shape, (1, SEQUENCE_LENGTH, self.embedding.embedding_size))


class CustomEmbeddingsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CustomEmbeddingsTest, self).__init__(*args, **kwargs)
        self.embedding = CustomEmbedding('empty_embedding',
                                         sequence_length=SEQUENCE_LENGTH,
                                         embedding_size=100)

    def test_build_word2idx(self):
        corpus = [['我', '们', '变', '而', '以', '书', '会', '友', '，', '以', '书', '结', '缘', '，',
                   '把', '欧', '美', '、', '港', '台', '流', '行', '的',
                   '食', '品', '类', '图', '谱', '、', '画', '册', '、',
                   '工', '具', '书', '汇', '集', '一', '堂', '。'],
                  ['为', '了', '跟', '踪', '国', '际', '最', '新', '食', '品',
                   '工', '艺', '、', '流', '行', '趋', '势', '，', '大', '量',
                   '搜', '集', '海', '外', '专', '业', '书', '刊', '资', '料',
                   '是', '提', '高', '技', '艺', '的', '捷', '径', '。'],
                  ['其', '中', '线', '装', '古', '籍', '逾', '千', '册',
                   '；', '民', '国', '出', '版', '物', '几', '百', '种',
                   '；', '珍', '本', '四', '册', '、', '稀', '见', '本',
                   '四', '百', '余', '册', '，', '出', '版', '时', '间',
                   '跨', '越', '三', '百', '余', '年', '。'],
                  ['有', '的', '古', '木', '交', '柯', '，',
                   '春', '机', '荣', '欣', '，', '从', '诗',
                   '人', '句', '中', '得', '之', '，', '而',
                   '入', '画', '中', '，', '观', '之', '令', '人', '心', '驰', '。', '我']]
        self.embedding.build_token2idx_dict(x_data=corpus, min_count=2)

    def test_build(self):
        self.test_build_word2idx()
        self.assertEqual(self.embedding.token_count, 33)
        self.assertTrue(all(isinstance(x, str) for x in self.embedding.token2idx.keys()))
        self.assertTrue(all(isinstance(x, int) for x in self.embedding.token2idx.values()))
        self.assertEqual(self.embedding.idx2token[0], k.PAD)
        self.assertEqual(self.embedding.idx2token[1], k.BOS)
        self.assertEqual(self.embedding.idx2token[2], k.EOS)
        self.assertEqual(self.embedding.idx2token[3], k.UNK)

    def test_tokenize(self):
        self.test_build_word2idx()
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        tokens = self.embedding.tokenize(sentence)

        logging.info('tokenize test: {} -> {}'.format(sentence, tokens))
        self.assertEqual(len(tokens), len(tokens))
        self.assertEqual(tokens[-1], 3, msg='check unk value')

        token_list = self.embedding.tokenize([sentence])
        self.assertEqual(len(token_list[0]), len(sentence))

    def test_embed(self):
        self.test_build_word2idx()
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed(sentence)
        logging.info('embed test: {} -> {}'.format(sentence, embedded_sentence))
        self.assertEqual(embedded_sentence.shape, (SEQUENCE_LENGTH, self.embedding.embedding_size))

        embedded_sentences = self.embedding.embed([sentence])
        self.assertEqual(embedded_sentences.shape, (1, SEQUENCE_LENGTH, self.embedding.embedding_size))


if __name__ == '__main__':
    unittest.main()
