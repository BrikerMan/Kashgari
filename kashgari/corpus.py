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
import logging
import pandas as pd

from typing import Tuple, List
from kashgari.utils import helper
from kashgari.utils import downloader


class Corpus(object):
    __corpus_name__ = ''
    __zip_file__name = ''

    __desc__ = ''

    @classmethod
    def get_classification_data(cls,
                                is_test: bool = False,
                                shuffle: bool = True,
                                max_count: int = 0) -> Tuple[List[str], List[str]]:
        raise NotImplementedError()

    @classmethod
    def get_info(cls):
        raise NotImplementedError()


class TencentDingdangSLUCorpus(Corpus):

    __corpus_name__ = 'task-slu-tencent.dingdang-v1.1'
    __zip_file__name = 'task-slu-tencent.dingdang-v1.1.tar.gz'

    __desc__ = """    Download from NLPCC 2018 Task4 dataset
    details: http://tcci.ccf.org.cn/conference/2018/taskdata.php
    The dataset adopted by this task is a sample of the real query log from a commercial
    task-oriented dialog system. The data is all in Chinese. The evaluation includes three
    domains, namely music, navigation and phone call. Within the dataset, an additional
    domain label ‘OTHERS’ is used to annotate the data not covered by the three domains. To
    simplify the task, we keep only the intents and the slots of high-frequency while ignoring
    others although they appear in the original data. The entire data can be seen as a stream
    of user queries ordered by time stamp. The stream is further split into a series of segments
    according to the gaps of time stamps between queries and each segment is denoted as a
    ‘session’. The contexts within a session are taken into consideration when a query within
    the session was annotated. Below are two example sessions with annotations.
    
    sample
    ```
    1 打电话 phone_call.make_a_phone_call 打电话
    1 我想听美观 music.play 我想听<song>美观</song>
    1 我想听什话 music.play 我想听<song>什话||神话</song>
    1 神话 music.play <song>神话</song>
    
    2 播放调频广播 OTHERS 播放调频广播
    2 给我唱一首一晃就老了 music.play 给我唱一首<song>一晃就老了</song>
    ```
    """

    @classmethod
    def get_info(cls):
        folder_path = downloader.download_if_not_existed('corpus/' + cls.__corpus_name__,
                                                         'corpus/' + cls.__zip_file__name, )
        logging.info("""{} info\n    dataset path: {}\n{}""".format(cls.__corpus_name__,
                                                                    folder_path,
                                                                    cls.__desc__))

    @classmethod
    def get_classification_data(cls,
                                is_test: bool = False,
                                shuffle: bool = True,
                                max_count: int = 0) -> Tuple[List[str], List[str]]:
        folder_path = downloader.download_if_not_existed('corpus/' + cls.__corpus_name__,
                                                         'corpus/' + cls.__zip_file__name, )
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
    TencentDingdangSLUCorpus.get_info()
    # TencentDingdangSLUCorpus.get_classification_data()
    # print(TencentDingdangSLUCorpus.get_classification_data(is_test=True))


