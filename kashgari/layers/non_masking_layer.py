# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: non_masking_layer.py
# time: 2019-05-23 14:05

import tensorflow as tf
from tensorflow.python import keras

L = keras.layers

if tf.test.is_gpu_available(cuda_only=True):
    L.LSTM = L.CuDNNLSTM


class NonMaskingLayer(L.Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


if __name__ == "__main__":
    print("Hello world")
