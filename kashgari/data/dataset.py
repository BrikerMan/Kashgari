# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: dataset.py
@time: 2019-01-19 17:36

"""
import os
import logging
import pandas as pd
import pickle
from typing import List, Tuple, Optional, Dict
from kashgari.data import data_reader
from kashgari import k
from kashgari.utils.downloader import download_if_not_existed


class Corpus(object):
    __file_name__ = ''
    __label_key__ = 'label'
    __text_key__ = 'review'

    __segment_method__ = k.SegmenterType.jieba

    @classmethod
    def __get_cached_file_path__(cls):
        cached_file_name = '{}_{}.pickle'.format(cls.__file_name__, cls.__segment_method__.value)
        return os.path.join(k.PROCESSED_CORPUS_PATH, cached_file_name)

    @classmethod
    def load(cls) -> Tuple[List[List[str]], List[str]]:
        data = cls.__load_cache_dataset__()
        if data:
            return data['x_data'], data['y_data']
        file_path = download_if_not_existed('dataset/' + cls.__file_name__)
        x_data, y_data = data_reader.load_data_from_csv_for_classifier(file_path,
                                                                       cls.__text_key__,
                                                                       cls.__label_key__,
                                                                       cls.__segment_method__)
        data = {
            'x_data': x_data,
            'y_data': y_data
        }

        with open(cls.__get_cached_file_path__(), 'wb') as f:
            pickle.dump(data, f)
        return x_data, y_data

    @classmethod
    def __load_cache_dataset__(cls) -> Optional[Dict]:
        try:
            cached_file = cls.__get_cached_file_path__()
            if os.path.exists(cached_file):
                with open(cached_file, 'rb') as f:
                    data = pickle.load(f)
                    assert len(data['x_data']) == len(data['y_data'])
                    return data
        except Exception as e:
            logging.error('read cached file failed, e: {}'.format(e))
            pass


class SimplifyWeibo4MoodsCorpus(Corpus):
    __file_name__ = 'simplify_weibo_4_moods.csv'

    __label_key__ = 'label'
    __text_key__ = 'review'

    __segment_method__ = k.SegmenterType.jieba


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger

    init_logger()
    x, y = SimplifyWeibo4MoodsCorpus.load()
    print(len(x))
    print(len(y))
    print("Hello world")
