# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: macros.py
@time: 2019-05-17 11:38

"""
import os
import logging
from pathlib import Path
import tensorflow as tf

DATA_PATH = os.path.join(str(Path.home()), '.kashgari')

Path(DATA_PATH).mkdir(exist_ok=True, parents=True)


class TaskType(object):
    CLASSIFICATION = 'classification'
    LABELING = 'labeling'
    SCORING = 'scoring'


class Config(object):

    def __init__(self):
        self._use_cudnn_cell = False
        self.disable_auto_summary = False

        if tf.test.is_gpu_available(cuda_only=True):
            logging.warning("CUDA GPU available, you can set `kashgari.config.use_cudnn_cell = True` to use CuDNNCell. "
                            "This will speed up the training, "
                            "but will make model incompatible with CPU device.")

    @property
    def use_cudnn_cell(self):
        return self._use_cudnn_cell

    @use_cudnn_cell.setter
    def use_cudnn_cell(self, value):
        self._use_cudnn_cell = value
        from kashgari.layers import L
        if value:
            if tf.test.is_gpu_available(cuda_only=True):
                L.LSTM = tf.compat.v1.keras.layers.CuDNNLSTM
                L.GRU = tf.compat.v1.keras.layers.CuDNNGRU
                logging.warning("CuDNN enabled, this will speed up the training, "
                                "but will make model incompatible with CPU device.")
            else:
                logging.warning("Unable to use CuDNN cell, no GPU available.")
        else:
            L.LSTM = tf.keras.layers.LSTM
            L.GRU = tf.keras.layers.GRU

    def to_dict(self):
        return {
            'use_cudnn_cell': self.use_cudnn_cell
        }


config = Config()

if __name__ == "__main__":
    print("Hello world")
