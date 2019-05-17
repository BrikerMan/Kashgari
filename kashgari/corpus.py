# encoding: utf-8

import os
import kashgari.macros as k
from typing import Tuple, List
from tensorflow.python.keras.utils import get_file


CORPUS_PATH = os.path.join(k.DATA_PATH, 'corpus')


class ChineseDailyNerCorpus(object):
    """
    Chinese Daily New New Corpus
    https://github.com/zjy-ucas/ChineseNER/
    """
    __corpus_name__ = 'china-people-daily-ner-corpus'
    __zip_file__name = 'http://storage.eliyar.biz/corpus/china-people-daily-ner-corpus.tar.gz'

    @classmethod
    def load_data(cls, subset_name: str = 'train') -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load dataset as sequence labeling format, char level tokenized

        features: [['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', ...], ...]

        labels: [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', ...], ...]

        Sample::

            x, y = ChineseDailyNerCorpus.load_data('train')
        Args:
            subset_name: {train, test, valid}

        Returns:
            dataset_features and dataset labels
        """
        corpus_path = get_file(cls.__corpus_name__,
                               cls.__zip_file__name,
                               cache_dir=k.DATA_PATH,
                               untar=True)

        if subset_name == 'train':
            file_path = os.path.join(corpus_path, 'example.train')
        elif subset_name == 'test':
            file_path = os.path.join(corpus_path, 'example.test')
        else:
            file_path = os.path.join(corpus_path, 'example.dev')

        data_x, data_y = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            x, y = [], []
            for line in lines:
                rows = line.split(' ')
                if len(rows) == 1:
                    data_x.append(x)
                    data_y.append(y)
                    x = []
                    y = []
                else:
                    x.append(rows[0])
                    y.append(rows[1])
        return data_x, data_y


if __name__ == "__main__":
    x, y = ChineseDailyNerCorpus.load_data('train')
    print(x[:1])
    print(y[:1])
    print("Hello world")
