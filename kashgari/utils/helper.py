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
import random

import h5py
import numpy as np
from keras.utils import to_categorical


def h5f_generator(h5path: str,
                  # indices: List[int],
                  num_classes: int,
                  batch_size: int = 128):
    """
    fit generator for h5 file
    :param h5path: target f5file
    :param num_classes: label counts to covert y label to one hot array
    :param batch_size:
    :return:
    """

    db = h5py.File(h5path, "r")
    while True:
        page_list = list(range(len(db['x']) // batch_size + 1))
        random.shuffle(page_list)
        for page in page_list:
            x = db["x"][page: (page + 1) * batch_size]
            y = to_categorical(db["y"][page: (page + 1) * batch_size],
                               num_classes=num_classes,
                               dtype=np.int)
            yield (x, y)


if __name__ == "__main__":
    print("Hello world")
