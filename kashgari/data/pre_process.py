# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: pre_process.py
@time: 2019-01-19 12:46

"""
import jieba
import pandas as pd


def pre_process_df(data_frame: pd.DataFrame,
                   text_column_label: str = 'review',
                   class_column_label: str = 'label'):
    df = pd.DataFrame()
    df['text'] = [' '.join(jieba.cut(text.strip())) for text in data_frame[text_column_label]]
    df['class'] = data_frame[class_column_label]
    return df


if __name__ == "__main__":
    df = pd.read_csv('/Users/brikerman/Downloads/simplifyweibo_4_moods.csv')
    df = df[:100]
    df = pre_process_df(df)
    df.to_csv('dataset.csv')
    print("Hello world")
