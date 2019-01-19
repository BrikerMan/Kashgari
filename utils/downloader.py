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
import os
import bz2
import download
from typing import Union
from utils.macros import DATA_PATH
from utils.macros import STORAGE_HOST
from utils.macros import Word2VecModels


URL_MAP = {
    Word2VecModels.sgns_weibo_bigram: 'embedding/word2vev/sgns.weibo.bigram.bz2'
}


def download_file(file: str):
    url = STORAGE_HOST + file
    target_path = os.path.join(DATA_PATH, file)
    download.download(url, target_path)


def download_if_not_existed(file: Union[Word2VecModels, str]):
    file_path = URL_MAP.get(file, file)
    target_path = os.path.join(DATA_PATH, file_path)
    if not os.path.exists(target_path[:-4]):
        download_file(file_path)
        with open(target_path, 'rb') as source, open(target_path[:-4], 'wb') as dest:
            dest.write(bz2.decompress(source.read()))


if __name__ == "__main__":
    from utils.logger import init_logger
    init_logger()
    download_if_not_existed(Word2VecModels.sgns_weibo_bigram)
    print("Hello world")
