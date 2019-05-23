# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 2019-05-23 14:05

import tensorflow as tf
from tensorflow.python import keras
from kashgari.layers.non_masking_layer import NonMaskingLayer
from kashgari.layers.crf import CRF, crf_accuracy

L = keras.layers

if tf.test.is_gpu_available(cuda_only=True):
    L.LSTM = L.CuDNNLSTM

if __name__ == "__main__":
    print("Hello world")