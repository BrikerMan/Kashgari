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
from typing import List, Tuple

from kashgari.data import data_reader
from kashgari.utils import k
from kashgari.utils.downloader import download_if_not_existed


class Corpus(object):
    __file_name__ = ''
    __label_key__ = 'label'
    __text_key__ = 'review'

    __segment_method__ = k.Segmenter.jieba

    @classmethod
    def load(cls) -> Tuple[List[List[str]], List[str]]:
        file_path = download_if_not_existed('dataset/' + cls.__file_name__)
        x_data, y_data = data_reader.load_data_from_csv_for_classifier(file_path,
                                                                       cls.__text_key__,
                                                                       cls.__label_key__,
                                                                       cls.__segment_method__)
        return x_data, y_data


class SimplifyWeibo4MoodsCorpus(Corpus):
    __file_name__ = 'simplify_weibo_4_moods.csv'

    __label_key__ = 'label'
    __text_key__ = 'review'

    __segment_method__ = k.Segmenter.jieba


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger

    init_logger()
    x, y = SimplifyWeibo4MoodsCorpus.load()
    print(x[:10])
    print(y[:10])
    print("Hello world")
