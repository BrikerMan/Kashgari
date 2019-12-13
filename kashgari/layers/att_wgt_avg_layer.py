# encoding: utf-8

# author: AlexWang
# contact: ialexwwang@gmail.com

# file: attention_weighted_average.py
# time: 2019-06-24 19:35

from tensorflow.python import keras
from tensorflow.python.keras import backend as K

import kashgari

L = keras.layers
initializers = keras.initializers
InputSpec = L.InputSpec


class AttentionWeightedAverageLayer(L.Layer):
    '''
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    '''

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2].value, 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init,
                                 trainable=True
                                 )
        # self.trainable_weights = [self.W]
        super(AttentionWeightedAverageLayer, self).build(input_shape)

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

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def get_config(self):
        config = {'return_attention': self.return_attention, }
        base_config = super(AttentionWeightedAverageLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


AttentionWeightedAverage = AttentionWeightedAverageLayer
AttWgtAvgLayer = AttentionWeightedAverageLayer

kashgari.custom_objects['AttentionWeightedAverageLayer'] = AttentionWeightedAverageLayer
kashgari.custom_objects['AttentionWeightedAverage'] = AttentionWeightedAverage
kashgari.custom_objects['AttWgtAvgLayer'] = AttWgtAvgLayer

if __name__ == '__main__':
    print('Hello world, AttentionWeightedAverageLayer/AttWgtAvgLayer.')
