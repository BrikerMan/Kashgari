# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: corpus.py
# time: 12:38 下午

import os
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import get_file

from kashgari import macros as K
from kashgari import utils
from kashgari.logger import logger
from kashgari.tokenizers.base_tokenizer import Tokenizer
from kashgari.tokenizers.bert_tokenizer import BertTokenizer

CORPUS_PATH = os.path.join(K.DATA_PATH, 'corpus')


class DataReader:

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
            x: List[str] = []
            y: List[str] = []
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


class ChineseDailyNerCorpus:
    """
    Chinese Daily New New Corpus
    https://github.com/zjy-ucas/ChineseNER/

    Example:
        >>> from kashgari.corpus import ChineseDailyNerCorpus
        >>> train_x, train_y = ChineseDailyNerCorpus.load_data('train')
        >>> test_x, test_y = ChineseDailyNerCorpus.load_data('test')
        >>> valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        >>> print(train_x)
            [['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', ...], ...]
        >>> print(train_y)
            [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', ...], ...]
    """
    __corpus_name__ = 'china-people-daily-ner-corpus'
    __zip_file__name = 'http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz'

    @classmethod
    def load_data(cls,
                  subset_name: str = 'train',
                  shuffle: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load dataset as sequence labeling format, char level tokenized

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
        logger.debug(f"loaded {len(x_data)} samples from {file_path}. Sample:\n"
                     f"x[0]: {x_data[0]}\n"
                     f"y[0]: {y_data[0]}")
        return x_data, y_data


class SMP2018ECDTCorpus:
    """
    https://worksheets.codalab.org/worksheets/0x27203f932f8341b79841d50ce0fd684f/

    This dataset is released by the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2018-ECDT)
    task 1 and is provided by the iFLYTEK Corporation, which is a Chinese human-computer dialogue dataset.

    Sample::

              label           query
        0   weather        今天东莞天气如何
        1       map  从观音桥到重庆市图书馆怎么走
        2  cookbook          鸭蛋怎么腌？
        3    health         怎么治疗牛皮癣
        4      chat             唠什么

    Example:
        >>> from kashgari.corpus import SMP2018ECDTCorpus
        >>> train_x, train_y = SMP2018ECDTCorpus.load_data('train')
        >>> test_x, test_y = SMP2018ECDTCorpus.load_data('test')
        >>> valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        >>> print(train_x)
        [['听', '新', '闻', '。'], ['电', '视', '台', '在', '播', '什', '么'], ...]
        >>> print(train_y)
        ['news', 'epg', ...]
    """

    __corpus_name__ = 'SMP2018ECDTCorpus'
    __zip_file__name = 'http://s3.bmio.net/kashgari/SMP2018ECDTCorpus.tar.gz'

    @classmethod
    def load_data(cls,
                  subset_name: str = 'train',
                  shuffle: bool = True,
                  cutter: str = 'char') -> Tuple[List[List[str]], List[str]]:
        """
        Load dataset as sequence classification format, char level tokenized

        Args:
            subset_name: {train, test, valid}
            shuffle: should shuffle or not, default True.
            cutter: sentence cutter, {char, jieba}

        Returns:
            dataset_features and dataset labels

        """

        corpus_path = get_file(cls.__corpus_name__,
                               cls.__zip_file__name,
                               cache_dir=K.DATA_PATH,
                               untar=True)

        if cutter not in ['char', 'jieba', 'none']:
            raise ValueError('cutter error, please use one onf the {char, jieba}')

        df_path = os.path.join(corpus_path, f'{subset_name}.csv')
        df = pd.read_csv(df_path)
        if cutter == 'jieba':
            try:
                import jieba
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "please install jieba, `$ pip install jieba`")
            x_data = [list(jieba.cut(item)) for item in df['query'].to_list()]
        elif 'char':
            x_data = [list(item) for item in df['query'].to_list()]
        y_data = df['label'].to_list()

        if shuffle:
            x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)
        logger.debug(f"loaded {len(x_data)} samples from {df_path}. Sample:\n"
                     f"x[0]: {x_data[0]}\n"
                     f"y[0]: {y_data[0]}")
        return x_data, y_data


class JigsawToxicCommentCorpus:
    """
    Kaggle Toxic Comment Classification Challenge corpus

    You need to download corpus from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview
    to a folder. Then init a JigsawToxicCommentCorpus object with `train.csv` path.

    Examples:

        >>> from kashgari.corpus import JigsawToxicCommentCorpus
        >>> corpus = JigsawToxicCommentCorpus('<train.csv file-path>')
        >>> train_x, train_y = corpus.load_data('train')
        >>> test_x, test_y = corpus.load_data('test')
        >>> print(train_x)
        [['Please', 'stop', 'being', 'a', 'penis—', 'and', 'Grow', 'Up', 'Regards-'], ...]
        >>> print(train_y)
        [['obscene', 'insult'], ...]
    """

    def __init__(self,
                 corpus_train_csv_path: str,
                 sample_count: int = None,
                 tokenizer: Tokenizer = None) -> None:
        self.file_path = corpus_train_csv_path
        self.train_ids = []
        self.test_ids = []
        self.valid_ids = []

        self.tokenizer: Tokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizer()
        else:
            self.tokenizer = tokenizer

        if sample_count is None:
            df = pd.read_csv(self.file_path)
            sample_count = len(df)
            del df
        self.sample_count = sample_count

        for i in range(self.sample_count):
            prob = np.random.random()
            if prob <= 0.7:
                self.train_ids.append(i)
            elif prob <= 0.85:
                self.test_ids.append(i)
            else:
                self.valid_ids.append(i)

    @classmethod
    def _extract_label(cls, row: pd.Series) -> List[str]:
        y = []
        for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            if row[label] == 1:
                y.append(label)
        return y

    def _text_process(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def load_data(self,
                  subset_name: str = 'train',
                  shuffle: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load dataset as sequence labeling format, char level tokenized

        Args:
           subset_name: {train, test, valid}
           shuffle: should shuffle or not, default True.

        Returns:
           dataset_features and dataset labels
        """

        df = pd.read_csv(self.file_path)
        df = df[:self.sample_count]
        df['y'] = df.apply(self._extract_label, axis=1)
        df['x'] = df['comment_text'].apply(self._text_process)
        df = df[['x', 'y']]
        if subset_name == 'train':
            df = df.loc[self.train_ids]
        elif subset_name == 'valid':
            df = df.loc[self.valid_ids]
        else:
            df = df.loc[self.test_ids]

        xs, ys = list(df['x'].values), list(df['y'].values)
        if shuffle:
            xs, ys = utils.unison_shuffled_copies(xs, ys)
        return xs, ys


if __name__ == "__main__":
    corpus = JigsawToxicCommentCorpus(
        '/Users/brikerman/Downloads/jigsaw-toxic-comment-classification-challenge/train.csv')
    x, y = corpus.load_data()
    for i in x[:20]:
        print(i)

    for i in y[:20]:
        print(i)
