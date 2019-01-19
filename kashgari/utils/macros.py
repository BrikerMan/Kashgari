# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: macros.py
@time: 2019-01-19 09:58

"""
import os
import bz2
import download
from typing import Union
from pathlib import Path
from enum import Enum

PAD = "#PAD"
BOS = "#BOS"
EOS = "#EOS"
UNK = "#UNK"

MARKED_KEYS = [PAD, BOS, EOS, UNK]

NO_TAG = 'O'

home = str(Path.home())

DATA_PATH = os.path.join(home, '.kashgari')
STORAGE_HOST = 'https://storage.eliyar.biz/'


class Word2VecModels(Enum):
    """
    provided pre trained word2vec from https://github.com/Embedding/Chinese-Word-Vectors
    """
    sgns_weibo_bigram = 'sgns.weibo.bigram.bz2'


URL_MAP = {
    Word2VecModels.sgns_weibo_bigram: 'embedding/word2vev/sgns.weibo.bigram.bz2'
}


def download_file(file: str):
    url = STORAGE_HOST + file
    target_path = os.path.join(DATA_PATH, file)
    download.download(url, target_path)


def download_if_not_existed(file_path: str) -> str:
    target_path = os.path.join(DATA_PATH, file_path)
    if not os.path.exists(target_path[:-4]):
        download_file(file_path)
        with open(target_path, 'rb') as source, open(target_path[:-4], 'wb') as dest:
            dest.write(bz2.decompress(source.read()))
    return target_path[:-4]


def get_model_path(file: Union[Word2VecModels, str]) -> str:
    file_path = URL_MAP.get(file, file)
    return download_if_not_existed(file_path)


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    init_logger()
