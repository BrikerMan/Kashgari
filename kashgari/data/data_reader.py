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
from typing import List, Tuple
import pandas as pd


def load_data_from_csv(file: str) -> Tuple[List[List[str]], List[int]]:
    df = pd.read_csv(file)
    data_x: List[List[str]] = [item.split(' ') for item in df['text']]
    data_y: List[int] = [int(item) for item in df['class']]
    return data_x, data_y


if __name__ == "__main__":
    file_path = '/Users/brikerman/Desktop/ailab/Kashgari/kashgari/data/dataset.csv'
    x, y = load_data_from_csv(file_path)
    print(x[:2])
    print(y[:2])
    print("Hello world")
