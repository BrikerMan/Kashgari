#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/8/29 11:11 上午
# File    : tools.py
# Project : Kashgari

import os
import zipfile
import pathlib
from tensorflow.keras.utils import get_file
from kashgari import macros as K


def get_bert_path() -> str:
    url = "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"
    bert_path = os.path.join(K.DATA_PATH, 'datasets', 'bert')
    model_path = os.path.join(bert_path, 'chinese_L-12_H-768_A-12')
    pathlib.Path(bert_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_path):
        zip_file_path = get_file("bert/chinese_L-12_H-768_A-12.zip",
                                 url,
                                 cache_dir=K.DATA_PATH, )

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(bert_path)
    return model_path


if __name__ == '__main__':
    for k, v in os.environ.items():
        print(f'{k:20}: {v}')
    get_bert_path()
