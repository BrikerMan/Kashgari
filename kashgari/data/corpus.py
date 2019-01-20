# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: corpus
@time: 2019-01-20

"""
import tqdm
import random
import logging
import numpy as np
import pandas as pd

from kashgari import k
from kashgari.tokenizer import Tokenizer
from kashgari.utils import helper
from kashgari.utils import downloader

from keras.preprocessing import sequence
from keras.utils import to_categorical


class Corpus(object):
    __file_name__ = ''
    __task_list__ = k.TaskType.classification

    __segmenter__ = k.SegmenterType.space
    __sequence_length__ = 30

    __class_map__ = {}

    def __init__(self,
                 tokenizer: Tokenizer,
                 data_count_limit: int = 0,
                 validate_data: int = 0.1,
                 test_data: int = 0.1,
                 **kwargs):
        self.tokenizer = tokenizer
        self.data_count_limit = data_count_limit
        self.tokenizer.segmenter = self.__segmenter__
        self.tokenizer.sequence_length = self.__sequence_length__

        self.x_data = []
        self.y_data = []

        self.validate_data = validate_data
        self.test_data = test_data
        self.build()

    @property
    def data_count(self):
        return len(self.x_data)

    def build(self):
        self.x_data, self.y_data = self.read_original_data()
        self.x_data, self.y_data = helper.unison_shuffled_copies(self.x_data, self.y_data)

        self.tokenizer.idx2label = self.__class_map__
        self.tokenizer.label2idx = dict([(value, key) for key, value in self.__class_map__.items()])

    @classmethod
    def get_recommend_tokenizer_config(cls):
        return {}

    def read_original_data(self):
        file_path = downloader.download_if_not_existed('corpus/' + self.__file_name__)
        df = pd.read_csv(file_path)
        return df['review'], df['label']

    def tokenize_and_process(self,
                             x_data,
                             y_data):
        raise NotImplementedError

    # def tokenize_samples(self):
    #     sample_index = random.sample(list(range(len(x_data))), 3)
    #
    #     logging.info('------------ Tokenizer pre-process completed -----------')
    #     for index in sample_index:
    #         logging.info('sample {:5} raw       : {} -> {}'.format(index,
    #                                                                x_data[index],
    #                                                                y_data[index]))
    #         logging.info('sample {:5} tokenized : {} -> {}'.format(index,
    #                                                                tokenized_x_data[index],
    #                                                                tokenized_y_data[index]))

    # def load(self,
    #          shuffle: bool = True):
    #     x_data, y_data = self.read_original_data()
    #
    #     if shuffle:
    #         x_data, y_data = helper.unison_shuffled_copies(x_data, y_data)
    #     if self.data_count_limit != 0:
    #         x_data = x_data[:self.data_count_limit]
    #         y_data = y_data[:self.data_count_limit]
    #
    #     x_data, y_data = self.preprocess(x_data, y_data)
    #     return x_data, y_data

    def fit_generator(self,
                      batch_size: int = 128,
                      dataset_type: k.DataSetType = k.DataSetType.train):
        validate_index = int(len(self.x_data) * self.validate_data)
        test_index = int(len(self.x_data) * self.test_data) + validate_index

        if dataset_type == k.DataSetType.train:
            x_data, y_data = self.x_data[:validate_index], self.y_data[:validate_index]
        elif dataset_type == k.DataSetType.test:
            x_data, y_data = self.x_data[validate_index:test_index], self.y_data[validate_index:test_index]
        else:
            x_data, y_data = self.x_data[test_index:], self.y_data[test_index:]

        while True:
            page_list = list(range(len(x_data) // batch_size + 1))
            random.shuffle(page_list)
            for page in page_list:
                target_x = x_data[page: (page + 1) * batch_size]
                target_y = y_data[page: (page + 1) * batch_size]
                x, y = self.tokenize_and_process(target_x, target_y)
                x = sequence.pad_sequences(x,
                                           maxlen=self.tokenizer.sequence_length)
                y = to_categorical(y,
                                   num_classes=self.tokenizer.class_num,
                                   dtype=np.int)
                yield (x, y)


class SimplifyWeibo4MoodsCorpus(Corpus):
    __file_name__ = 'simplify_weibo_4_moods.csv'
    __task_list__ = k.TaskType.classification

    __segmenter__ = k.SegmenterType.jieba
    __sequence_length__ = 50
    __class_map__ = {
        0: '喜悦',
        1: '愤怒',
        2: '厌恶',
        3: '低落'
    }

    @classmethod
    def get_recommend_tokenizer_config(cls):
        return {
            'embedding_name': k.Word2VecModels.sgns_weibo_bigram,
            'sequence_length': 50,
            'segmenter': k.SegmenterType.jieba.value
        }

    def read_original_data(self):
        file_path = downloader.download_if_not_existed('corpus/' + self.__file_name__)
        df = pd.read_csv(file_path)
        return df['review'], df['label']

    def tokenize_and_process(self,
                             x_data,
                             y_data):
        tokenized_x_data = []
        for x_item in tqdm.tqdm(x_data, desc='Tokenizer tokenizing x_data'):
            tokenized_x_data.append(self.tokenizer.word_to_token(x_item))

        tokenized_y_data = []
        for y_item in tqdm.tqdm(y_data, desc='Tokenizer tokenizing y_data'):
            tokenized_y_data.append(self.tokenizer.label_to_token(y_item))
        return tokenized_x_data, tokenized_y_data


if __name__ == '__main__':
    from kashgari.utils.logger import init_logger
    init_logger()
    tokenizer_obj = Tokenizer(k.Word2VecModels.sgns_weibo_bigram)
    corpus = SimplifyWeibo4MoodsCorpus(data_count_limit=1000,
                                       tokenizer=tokenizer_obj)
    x, y = corpus.fit_generator()
    logging.info(x[:3])
    logging.info(y[:3])
    print("hello, world")
