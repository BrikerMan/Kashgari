# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: data_reader.py
@time: 2019-01-19 12:41

"""
from typing import List

import pandas as pd
import tqdm

from kashgari import k, ClassificationData


def load_data_from_csv_for_classifier(csv_path: str,
                                      text_column_label: str = 'review',
                                      class_column_label: str = 'label',
                                      token_method: k.Segmenter = k.Segmenter.space) -> ClassificationData:
    df = pd.read_csv(csv_path)
    data_x: List[List[str]] = []
    for text in tqdm.tqdm(df[text_column_label], desc="preparing data with {} segmenter".format(token_method.value)):
        data_x.append(tokenize_func(text, token_method))
    data_y: List[str] = df[class_column_label]
    return data_x, data_y


def tokenize_func(text: str,
                  method: k.Segmenter = k.Segmenter.space):
    if method == k.Segmenter.jieba:
        import jieba
        return list(jieba.cut(text.strip()))
    else:
        return text.split(' ')


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger

    init_logger()
    file_path = '/Users/brikerman/Desktop/ailab/Kashgari/kashgari/data/dataset.csv'
    x, y = load_data_from_csv_for_classifier(file_path)
    print(x[:2])
    print(y[:2])
    print("Hello world")
