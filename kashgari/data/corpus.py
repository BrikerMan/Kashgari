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
    __corpus_name__ = ''
    __corpus_desc__ = ''
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
        all_data_count = len(self.x_data)
        self.x_data, self.y_data = helper.unison_shuffled_copies(self.x_data, self.y_data)
        if self.data_count_limit:
            self.x_data = self.x_data[:self.data_count_limit]
            self.y_data = self.y_data[:self.data_count_limit]

        self.tokenizer.idx2label = self.__class_map__
        self.tokenizer.label2idx = dict([(value, key) for key, value in self.__class_map__.items()])

        assert len(self.x_data) == len(self.y_data)
        logging.info('------------- corpus load finished -----------------')
        logging.info('corpus name : {}'.format(self.__corpus_name__))
        logging.info('corpus desc : {}'.format(self.__corpus_desc__))
        logging.info('total data  : {}'.format(all_data_count))
        logging.info('loaded data : {}'.format(len(self.x_data)))
        self.calculate_class_info()

    @classmethod
    def get_recommend_tokenizer_config(cls):
        return {}

    def pre_process_and_cache(self):
        pass

    def calculate_class_info(self):
        label2count = {}
        for y in self.y_data:
            label2count[y] = label2count.get(y, 0) + 1
        logging.info('class count : {}'.format(label2count))

    def read_original_data(self):
        file_path = downloader.download_if_not_existed('corpus/' + self.__file_name__)
        df = pd.read_csv(file_path)
        return df['review'], df['label']

    def tokenize_and_process(self,
                             x_data,
                             y_data):
        raise NotImplementedError

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
                print(x.shape)
                print(y.shape)
                yield (x, y)


class SimplifyWeibo4MoodsCorpus(Corpus):
    __corpus_name__ = 'Simplify Weibo 4 Moods Corpus'
    __corpus_desc__ = 'Weibo corpus with 4 simple mood class'

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
            'segmenter': cls.__segmenter__
        }

    def read_original_data(self):
        file_path = downloader.download_if_not_existed('corpus/' + self.__file_name__)
        df = pd.read_csv(file_path)
        return df['review'], df['label']

    def tokenize_and_process(self,
                             x_data,
                             y_data):
        tokenized_x_data = []
        for x_item in x_data:
            tokenized_x_data.append(self.tokenizer.word_to_token(x_item))

        tokenized_y_data = []
        for y_item in y_data:
            tokenized_y_data.append(self.tokenizer.label_to_token(y_item))
        return tokenized_x_data, tokenized_y_data


if __name__ == '__main__':
    from kashgari.utils.logger import init_logger
    init_logger()
    tokenizer_obj = Tokenizer(k.Word2VecModels.sgns_weibo_bigram)
    corpus = SimplifyWeibo4MoodsCorpus(data_count_limit=1000,
                                       tokenizer=tokenizer_obj)
    print(corpus.x_data[:10])
    print(corpus.y_data[:10])
    generator = corpus.fit_generator()
    logging.info(next(generator))
    logging.info(next(generator))
    logging.info(next(generator))

