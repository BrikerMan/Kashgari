# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: custom_generator.py
# time: 4:13 下午

import os
import linecache
from tensorflow.keras.utils import get_file
from kashgari.generators import ABCGenerator


def download_data(duplicate=1000):
    url_list = [
        'https://raw.githubusercontent.com/BrikerMan/JointSLU/master/data/atis-2.train.w-intent.iob',
        'https://raw.githubusercontent.com/BrikerMan/JointSLU/master/data/atis-2.dev.w-intent.iob',
        'https://raw.githubusercontent.com/BrikerMan/JointSLU/master/data/atis.test.w-intent.iob',
        'https://raw.githubusercontent.com/BrikerMan/JointSLU/master/data/atis.train.w-intent.iob'
    ]
    files = []
    for url in url_list:
        files.append(get_file(url.split('/')[-1], url))

    return files * duplicate


class ClassificationGenerator:
    def __init__(self, files):
        self.files = files
        self._line_count = sum(sum(1 for line in open(file, 'r')) for file in files)

    @property
    def steps(self) -> int:
        return self._line_count

    def __iter__(self):
        for file in self.files:
            with open(file, 'r') as f:
                for line in f:
                    rows = line.split('\t')
                    x = rows[0].strip().split(' ')[1:-1]
                    y = rows[1].strip().split(' ')[-1]
                    yield x, y


class LabelingGenerator(ABCGenerator):
    def __init__(self, files):
        self.files = files
        self._line_count = sum(sum(1 for line in open(file, 'r')) for file in files)

    @property
    def steps(self) -> int:
        return self._line_count

    def __iter__(self):
        for file in self.files:
            with open(file, 'r') as f:
                for line in f:
                    rows = line.split('\t')
                    x = rows[0].strip().split(' ')[1:-1]
                    y = rows[1].strip().split(' ')[1:-1]
                    yield x, y


def run_classification_model():
    from kashgari.tasks.classification import BiGRU_Model
    files = download_data()
    gen = ClassificationGenerator(files)

    model = BiGRU_Model()
    model.fit_generator(gen)


def run_labeling_model():
    from kashgari.tasks.labeling import BiGRU_Model
    files = download_data()
    gen = LabelingGenerator(files)

    model = BiGRU_Model()
    model.fit_generator(gen)


if __name__ == "__main__":
    run_classification_model()
