# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: layers
@time: 2019-02-23

"""
from __future__ import absolute_import, division
import logging

import tensorflow as tf
from keras.layers import Flatten
from keras.layers import GRU, LSTM
from keras.layers import CuDNNGRU, CuDNNLSTM
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K

from kashgari.macros import config

if config.use_CuDNN_cell:
    GRULayer = CuDNNGRU
    LSTMLayer = CuDNNLSTM
else:
    GRULayer = GRU
    LSTMLayer = LSTM


class AttentionWeightedAverage(Layer):
    '''
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    '''

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


class KMaxPooling(Layer):
    '''
    K-max pooling layer that extracts the k-highest activation from a sequence (2nd dimension).
    TensorFlow backend.

    # Arguments
        k: An int scale,
            indicate k max steps of features to pool.
        sorted: A bool,
            if output is sorted (default) or not.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, steps, features)` while `channels_first`
            corresponds to inputs with shape
            `(batch, features, steps)`.
    # Input shape
        - If `data_format='channels_last'`:
            3D tensor with shape:
            `(batch_size, steps, features)`
        - If `data_format='channels_first'`:
            3D tensor with shape:
            `(batch_size, features, steps)`
    # Output shape
        3D tensor with shape:
        `(batch_size, top-k-steps, features)`
    '''

    def __init__(self, k=1, sorted=True, data_format='channels_last', **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.sorted = sorted
        self.data_format = K.normalize_data_format(data_format)

    # def build(self, input_shape):
    #     assert len(input_shape) == 3
    #     super(KMaxPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], self.k, input_shape[1])
        else:
            return (input_shape[0], self.k, input_shape[2])

    def call(self, inputs):
        if self.data_format == 'channels_last':
            # swap last two dimensions since top_k will be applied along the last dimension
            shifted_input = tf.transpose(inputs, [0, 2, 1])

            # extract top_k, returns two tensors [values, indices]
            top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=self.sorted)[0]
        else:
            top_k = tf.nn.top_k(inputs, k=self.k, sorted=self.sorted)[0]
        # return flattened output
        return tf.transpose(top_k, [0, 2, 1])

    def get_config(self):
        config = {'k': self.k,
                  'sorted': self.sorted,
                  'data_format': self.data_format}
        base_config = super(KMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NonMaskingLayer(Layer):
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


if __name__ == '__main__':
    print("hello, world")
