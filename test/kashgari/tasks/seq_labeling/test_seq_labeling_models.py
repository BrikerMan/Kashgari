# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_lstm_model_test
@time: 2019-01-24

"""
# import logging
# import tempfile
# import unittest
#
# from kashgari.embeddings import WordEmbeddings, BERTEmbedding, BaseEmbedding
# from kashgari.tasks.seq_labeling import CNNLSTMModel, BLSTMModel, BLSTMCRFModel
# from kashgari.utils.logger import init_logger
#
# init_logger()
#
#
# class CNNLSTMModelTest(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super(CNNLSTMModelTest, self).__init__(*args, **kwargs)
#         self.x_data = [
#             '我 们 变 而 以 书 会 友 ， 以 书 结 缘 ， 把 欧 美 、'
#             ' 港 台 流 行 的 食 品 类 图 谱 、 画 册 、 工 具 书 汇 集 一 堂 。',
#             '为 了 跟 踪 国 际 最 新 食 品 工 艺 、 流 行 趋 势 ， 大 量 搜 集 '
#             '海 外 专 业 书 刊 资 料 是 提 高 技 艺 的 捷 径 。',
#             '其 中 线 装 古 籍 逾 千 册 ； 民 国 出 版 物 几 百 种 ； '
#             '珍 本 四 册 、 稀 见 本 四 百 余 册 ， 出 版 时 间 跨 越 三 百 余 年 。'
#         ]
#
#         self.y_data = [
#             'O O O O O O O O O O O O O O O B-LOC B-LOC O '
#             'B-LOC B-LOC O O O O O O O O O O O O O O O O O O O O',
#             'O O O O O O O O O O O O O O O O O O O O O O O O O O O'
#             ' O O O O O O O O O O O O',
#             'O O O O O O O O O O O O O O O O O O O O O O O O O O O'
#             ' O O O O O O O O O O O O O O O O'
#         ]
#
#         self.__model_class__ = CNNLSTMModel
#
#         self.x_data = [item.split(' ') for item in self.x_data]
#         self.y_data = [item.split(' ') for item in self.y_data]
#
#         self.x_eval = self.x_data
#         self.y_eval = self.y_data
#
#     def prepare_model(self, embedding: BaseEmbedding = None):
#         self.model = self.__model_class__(embedding)
#
#     def test_build(self):
#         self.prepare_model()
#         self.model.fit(self.x_data, self.y_data)
#         self.assertEqual(len(self.model.label2idx), 5)
#         self.assertGreater(len(self.model.token2idx), 4)
#         logging.info(self.model.embedding.token2idx)
#
#     def test_fit(self):
#         self.prepare_model()
#         self.model.fit(self.x_data, self.y_data, x_validate=self.x_eval, y_validate=self.y_eval)
#
#     def test_label_token_convert(self):
#         self.prepare_model()
#         self.test_fit()
#         sentence = list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分')
#         idxs = self.model.embedding.tokenize(sentence)
#         self.assertEqual(min(len(sentence), self.model.embedding.sequence_length),
#                          min(len(idxs)-2, self.model.embedding.sequence_length))
#         tokens = self.model.embedding.tokenize(sentence)
#         self.assertEqual(len(sentence)+2, len(tokens))
#
#     def test_predict(self):
#         self.test_fit()
#         sentence = list('语言学包含了几种分支领域。')
#         result = self.model.predict(sentence)
#         logging.info('test predict: {} -> {}'.format(sentence, result))
#         self.assertTrue(isinstance(self.model.predict(sentence)[0], str))
#         self.assertTrue(isinstance(self.model.predict([sentence])[0], list))
#         self.assertEqual(len(self.model.predict(sentence)), len(sentence))
#
#     def test_eval(self):
#         self.test_fit()
#         self.model.evaluate(self.x_data, self.y_data)
#
#     def test_bert(self):
#         embedding = BERTEmbedding('chinese_L-12_H-768_A-12', sequence_length=30)
#         self.prepare_model(embedding)
#
#         self.model.fit(self.x_data, self.y_data, x_validate=self.x_eval, y_validate=self.y_eval)
#         sentence = list('语言学包含了几种分支领域。')
#         logging.info(self.model.embedding.tokenize(sentence))
#         logging.info(self.model.predict(sentence))
#         self.assertTrue(isinstance(self.model.predict(sentence)[0], str))
#         self.assertTrue(isinstance(self.model.predict([sentence][0]), list))
#
#     def test_word2vec_embedding(self):
#         embedding = WordEmbeddings('sgns.weibo.bigram', sequence_length=30, limit=5000)
#         self.prepare_model(embedding)
#
#         self.model.fit(self.x_data, self.y_data, x_validate=self.x_eval, y_validate=self.y_eval)
#         sentence = list('语言学包含了几种分支领域。')
#         logging.info(self.model.embedding.tokenize(sentence))
#         logging.info(self.model.predict(sentence))
#         self.assertTrue(isinstance(self.model.predict(sentence)[0], str))
#         self.assertTrue(isinstance(self.model.predict([sentence][0]), list))
#
#     def test_save_and_load(self):
#         self.test_fit()
#         model_path = tempfile.gettempdir()
#         self.model.save(model_path)
#         new_model = BLSTMModel.load_model(model_path)
#         self.assertIsNotNone(new_model)
#         sentence = list('语言学包含了几种分支领域。')
#         result = new_model.predict(sentence)
#         self.assertTrue(isinstance(result[0], str))
#         self.assertEqual(len(sentence), len(result))
#
#
# class BLSTMModelTest(CNNLSTMModelTest):
#     def __init__(self, *args, **kwargs):
#         super(BLSTMModelTest, self).__init__(*args, **kwargs)
#         self.__model_class__ = BLSTMModel
#
#     def test_save_and_load(self):
#         super(BLSTMModelTest, self).test_save_and_load()
#
#
# class BLSTMCRFModelTest(CNNLSTMModelTest):
#     def __init__(self, *args, **kwargs):
#         super(BLSTMCRFModelTest, self).__init__(*args, **kwargs)
#         self.__model_class__ = BLSTMCRFModel
#
#     def test_save_and_load(self):
#         super(BLSTMCRFModelTest, self).test_save_and_load()
#
#
# if __name__ == "__main__":
#     unittest.main()
