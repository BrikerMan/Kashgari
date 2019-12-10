# encoding: utf-8

# author: AlexWang
# contact: ialexwwang@gmail.com

# file: attention_weighted_average.py
# time: 2019-06-25 16:35

import kashgari
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

L = keras.layers
InputSpec = L.InputSpec


class KMaxPoolingLayer(L.Layer):
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

    def __init__(self, k=1, sorted=True, data_format='channels_last', **kwargs):  # noqa: A002
        super(KMaxPoolingLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.sorted = sorted
        if data_format.lower() in ['channels_first', 'channels_last']:
            self.data_format = data_format.lower()
        else:
            self.data_format = K.image_data_format()

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
        base_config = super(KMaxPoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


KMaxPooling = KMaxPoolingLayer
KMaxPoolLayer = KMaxPoolingLayer

kashgari.custom_objects['KMaxPoolingLayer'] = KMaxPoolingLayer
kashgari.custom_objects['KMaxPooling'] = KMaxPooling
kashgari.custom_objects['KMaxPoolLayer'] = KMaxPoolLayer

if __name__ == '__main__':
    print('Hello world, KMaxPoolLayer/KMaxPoolingLayer.')

