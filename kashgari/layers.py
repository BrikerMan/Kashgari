# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: layers.py
# time: 2019-05-21 18:55

import tensorflow as tf
from tensorflow.python import keras

L = keras.layers

if tf.test.is_gpu_available(cuda_only=True):
    L.LSTM = L.CuDNNLSTM
else:
    L.LSTM = L.LSTM

if __name__ == "__main__":
    print("Hello world")
