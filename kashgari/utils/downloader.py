# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: downloader.py
@time: 2019-01-19 10:30

"""
import bz2
import os
from typing import Union

import download

from kashgari.macros import DATA_PATH
from kashgari.macros import STORAGE_HOST
from kashgari.macros import Word2VecModels

URL_MAP = {
    'sgns.weibo.bigram': 'embedding/word2vev/sgns.weibo.bigram.bz2'
}


def download_file(file_path: str, file: str):
    url = STORAGE_HOST + file
    file_path = os.path.join(DATA_PATH, file_path)
    download.download(url, os.path.dirname(file_path), kind='tar.gz', replace=True)
    # download.download(url, file_path)


def download_if_not_existed(path_or_name: str, zip_file_name: str) -> str:
    if not zip_file_name:
        zip_file_name = path_or_name

    if os.path.exists(path_or_name):
        return path_or_name
    elif os.path.exists(os.path.join(DATA_PATH, path_or_name)):
        return os.path.join(DATA_PATH, path_or_name)
    else:
        file_path = URL_MAP.get(path_or_name, path_or_name)
        download_file(file_path, zip_file_name)
        return os.path.join(DATA_PATH, path_or_name)


def get_cached_data_path(file: Union[Word2VecModels, str]) -> str:
    file_path = URL_MAP.get(file, file)
    target_path = os.path.join(DATA_PATH, 'pre_processed', file_path)
    return target_path


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    init_logger()
    download_if_not_existed(Word2VecModels.sgns_weibo_bigram)
    print("Hello world")
