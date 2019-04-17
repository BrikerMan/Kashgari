# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: helper.py
@time: 2019-01-19 16:25

"""
import logging
import os
import random
from typing import List, Optional

import download
import h5py
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.preprocessing import sequence
from keras.utils import to_categorical

from kashgari.macros import DATA_PATH
from kashgari.macros import STORAGE_HOST


# def h5f_generator(h5path: str,
#                   # indices: List[int],
#                   num_classes: int,
#                   batch_size: int = 128):
#     """
#     fit generator for h5 file
#     :param h5path: target f5file
#     :param num_classes: label counts to covert y label to one hot array
#     :param batch_size:
#     :return:
#     """
#
#     db = h5py.File(h5path, "r")
#     while True:
#         page_list = list(range(len(db['x']) // batch_size + 1))
#         random.shuffle(page_list)
#         for page in page_list:
#             x = db["x"][page: (page + 1) * batch_size]
#             y = to_categorical(db["y"][page: (page + 1) * batch_size],
#                                num_classes=num_classes,
#                                dtype=np.int)
#             yield (x, y)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def cached_path(file_path: str, download_url: Optional[str], sub_folders: List[str] = []):
    if os.path.exists(file_path):
        return file_path

    file_name_list = [DATA_PATH] + sub_folders + [file_path]
    file_path = os.path.join(*file_name_list)
    if os.path.exists(file_path):
        return file_path

    if download_url.startswith('http'):
        url = download_url
    else:
        url = STORAGE_HOST + download_url

    final_path = file_path
    if url.endswith('zip'):
        kind = 'zip'
        download_path = os.path.dirname(file_path)
    elif url.endswith('tar.gz'):
        kind = 'tar.gz'
        download_path = os.path.dirname(file_path)
    else:
        kind = 'file'
        download_path = file_path
    # url = url.replace('https://', 'http://')
    logging.info('start downloading file, if it takes too long, you could download with other downloader\n'
                 'url  : {}\n'
                 'path : {}'.format(url, file_path))
    e_path = download.download(url, download_path, kind=kind, replace=True)
    logging.info('downloader file_path {}, {} '.format(e_path, file_path))
    return final_path
    # if file_path.endswith('.bz2'):
    #     archive_path = e_path
    #     outfile_path = e_path[:-4]
    #     with open(archive_path, 'rb') as source, open(outfile_path, 'wb') as dest:
    #         dest.write(bz2.decompress(source.read()))
    #     return outfile_path
    # else:
    #     return final_path


# def check_should_download(file: str,
#                           download_url: Optional[str],
#                           sub_folders: List[str] = None):
#     """
#     check should download the file, if exist return file url, if not download and unzip
#     :param file:
#     :param sub_folders:
#     :param download_url:
#     :return:
#     """
#     logging.debug('check_should_download: file {}\ndownload_url {}\nsub_folders {}'.format(file,
#                                                                                            download_url,
#                                                                                            sub_folders))
#     if sub_folders is None:
#         sub_folders = []
#
#     if os.path.exists(file):
#         return file
#
#     folders = [DATA_PATH] + sub_folders + [file]
#     target_path = os.path.join(*folders)
#     original_file_path = target_path
#
#     if os.path.exists(target_path):
#         return target_path
#
#     if not download_url:
#         raise ValueError("need to provide valid model name or path")
#
#     if download_url.startswith('http'):
#         url = download_url
#     else:
#         url = STORAGE_HOST + download_url
#
#     if url.endswith('zip'):
#         kind = 'zip'
#     elif url.endswith('tar.gz'):
#         kind = 'tar.gz'
#     else:
#         kind = 'file'
#         target_path = os.path.join(target_path, url.split('/')[-1])
#
#     logging.info('start downloading file, if it takes too long, you could download with other downloader\n'
#                  'url  : {}\n'
#                  'path : {}'.format(url,
#                                     os.path.dirname(target_path)))
#
#     file_path = download.download(url, target_path, kind=kind, replace=True)
#     logging.debug('file downloaded to {}'.format(file_path))
#     if file_path.endswith('.bz2'):
#         archive_path = file_path
#         outfile_path = file_path[:-4]
#         with open(archive_path, 'rb') as source, open(outfile_path, 'wb') as dest:
#             dest.write(bz2.decompress(source.read()))
#         return original_file_path
#     else:
#         return target_path


def depth_count(lst: List[List]) -> int:
    return 1 + max(map(depth_count, lst)) if lst and isinstance(lst, list) else 0


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    init_logger()
    file = 'embedding/word2vec/sgns.weibo.bigram-char'
    url = 'embedding/word2vec/sgns.weibo.bigram-char.bz2'
    print(cached_path(file, url))
