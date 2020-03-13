# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: corpus.py
# time: 12:38 下午

import os
import logging
from kashgari import macros as K
from typing import Tuple, List
from tensorflow.keras.utils import get_file
from kashgari import utils

CORPUS_PATH = os.path.join(K.DATA_PATH, 'corpus')


class DataReader(object):

    @staticmethod
    def read_conll_format_file(file_path: str,
                               text_index: int = 0,
                               label_index: int = 1) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Read conll format data_file
        Args:
            file_path: path of target file
            text_index: index of text data, default 0
            label_index: index of label data, default 1

        Returns:

        """
        x_data, y_data = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            x, y = [], []
            for line in lines:
                rows = line.split(' ')
                if len(rows) == 1:
                    x_data.append(x)
                    y_data.append(y)
                    x = []
                    y = []
                else:
                    x.append(rows[text_index])
                    y.append(rows[label_index])
        return x_data, y_data


class ChineseDailyNerCorpus(object):
    """
    Chinese Daily New New Corpus
    https://github.com/zjy-ucas/ChineseNER/
    """
    __corpus_name__ = 'china-people-daily-ner-corpus'
    __zip_file__name = 'http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz'

    @classmethod
    def load_data(cls,
                  subset_name: str = 'train',
                  shuffle: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load dataset as sequence labeling format, char level tokenized

        features: ``[['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', ...], ...]``

        labels: ``[['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', ...], ...]``

        Sample::

            train_x, train_y = ChineseDailyNerCorpus.load_data('train')
            test_x, test_y = ChineseDailyNerCorpus.load_data('test')

        Args:
            subset_name: {train, test, valid}
            shuffle: should shuffle or not, default True.

        Returns:
            dataset_features and dataset labels
        """
        corpus_path = get_file(cls.__corpus_name__,
                               cls.__zip_file__name,
                               cache_dir=K.DATA_PATH,
                               untar=True)

        if subset_name == 'train':
            file_path = os.path.join(corpus_path, 'example.train')
        elif subset_name == 'test':
            file_path = os.path.join(corpus_path, 'example.test')
        else:
            file_path = os.path.join(corpus_path, 'example.dev')

        x_data, y_data = DataReader.read_conll_format_file(file_path)
        if shuffle:
            x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)
        logging.debug(f"loaded {len(x_data)} samples from {file_path}. Sample:\n"
                      f"x[0]: {x_data[0]}\n"
                      f"y[0]: {y_data[0]}")
        return x_data, y_data


if __name__ == "__main__":
    x, y = ChineseDailyNerCorpus.load_data()
    print(x)
