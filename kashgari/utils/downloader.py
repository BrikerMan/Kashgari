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
from typing import Union, Optional

import download

from kashgari.macros import DATA_PATH
from kashgari.macros import STORAGE_HOST

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

def check_should_download(file: str, download_url: Optional[str], unzip: bool = True):
    """check should download the file, if exist return file url, if not download and unzip
    
    Arguments:
        file {str} -- [description]
        download_url {Optional[str]} -- [description]
    
    Keyword Arguments:
        unzip {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """

    if os.path.exists(file):
        return file
    target_path = os.path.join(DATA_PATH, file)
    
    if os.path.exists(target_path):
        return target_path
    
    if download_url.startswith('http'):
        url = download_url
    else:
        url = STORAGE_HOST + url
    kind = 'file'
    if url.endswith('zip'):
        kind = 'zip'
    elif url.endswith('tar.gz'):
        kind = 'tar.gz'
    download.download(url, os.path.dirname(target_path), kind=kind, replace=True)


def get_cached_data_path(file: str) -> str:
    file_path = URL_MAP.get(file, file)
    target_path = os.path.join(DATA_PATH, 'pre_processed', file_path)
    return target_path


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    init_logger()
    check_should_download(file='uncased_L-12_H-768_A-12', download_url='https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
