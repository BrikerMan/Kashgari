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
import os
import pandas as pd

from typing import Tuple, List
from kashgari.utils import helper
from kashgari.utils import downloader


class Corpus(object):
    __file_name__ = ''
    __zip_file__name = ''

    @classmethod
    def get_classification_data(cls,
                                is_test: bool = False,
                                shuffle: bool = True,
                                max_count: int = 0) -> Tuple[List[str], List[str]]:
        raise NotImplementedError()


class TencentDingdangSLUCorpus(Corpus):

    __file_name__ = 'task-slu-tencent.dingdang-v1.1'
    __zip_file__name = 'task-slu-tencent.dingdang-v1.1.tar.gz'

    @classmethod
    def get_classification_data(cls,
                                is_test: bool = False,
                                shuffle: bool = True,
                                max_count: int = 0) -> Tuple[List[str], List[str]]:
        folder_path = downloader.download_if_not_existed('corpus/' + cls.__file_name__,
                                                         'corpus/' + cls.__zip_file__name,)
        if is_test:
            file_path = os.path.join(folder_path, 'test.csv')
        else:
            file_path = os.path.join(folder_path, 'train.csv')
        df = pd.read_csv(file_path)
        x_data = df['text'].values
        y_data = df['domain'].values
        if shuffle:
            x_data, y_data = helper.unison_shuffled_copies(x_data, y_data)
        if max_count != 0:
            x_data = x_data[:max_count]
            y_data = y_data[:max_count]
        return x_data, y_data


if __name__ == '__main__':
    from kashgari.utils.logger import init_logger
    init_logger()
    TencentDingdangSLUCorpus.get_classification_data()
    print(TencentDingdangSLUCorpus.get_classification_data(test=True))


