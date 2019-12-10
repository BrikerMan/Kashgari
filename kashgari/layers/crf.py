# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: crf.py
# time: 2019-06-28 14:33

import tensorflow as tf


class CRF(tf.keras.layers.Layer):
    """
        Conditional Random Field layer (tf.keras)
        `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
        must be equal to the number of classes the CRF can predict (a linear layer is recommended).
        Note: the loss and accuracy functions of networks using `CRF` must
        use the provided loss and accuracy functions (denoted as loss and viterbi_accuracy)
        as the classification of sequences are used with the layers internal weights.
        Args:
            output_dim (int): the number of labels to tag each temporal input.
        Input shape:
            nD tensor with shape `(batch_size, sentence length, num_classes)`.
        Output shape:
            nD tensor with shape: `(batch_size, sentence length, num_classes)`.
        """

    def __init__(self,
                 output_dim,
                 mode='reg',
                 supports_masking=False,
                 transitions=None,
                 **kwargs):
        self.transitions = None
        super(CRF, self).__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.mode = mode
        if self.mode == 'pad':
            self.input_spec = [tf.keras.layers.InputSpec(min_ndim=3), tf.keras.layers.InputSpec(min_ndim=2)]
        elif self.mode == 'reg':
            self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        else:
            raise ValueError
        self.supports_masking = supports_masking
        self.sequence_lengths = None

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'mode': self.mode,
            'supports_masking': self.supports_masking,
            'transitions': tf.keras.backend.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == 'pad':
            assert len(input_shape) == 2
            assert len(input_shape[0]) == 3
            assert len(input_shape[1]) == 2
            f_shape = tf.TensorShape(input_shape[0])
            input_spec = [tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]}),
                          tf.keras.layers.InputSpec(min_ndim=2, axes={-1: 1}, dtype=tf.int32)]
        else:
            assert len(input_shape) == 3
            f_shape = tf.TensorShape(input_shape)
            input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output shape. '
                             'Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        if self.mode == 'pad':
            sequences = tf.convert_to_tensor(inputs[0], dtype=self.dtype)
            self.sequence_lengths = tf.keras.backend.flatten(inputs[-1])
        else:
            sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
            shape = tf.shape(inputs)
            self.sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
        viterbi_sequence, _ = tf.contrib.crf.crf_decode(sequences, self.transitions,
                                                        self.sequence_lengths)
        output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
        return tf.keras.backend.in_train_phase(sequences, output)

    def loss(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        log_likelihood, self.transitions = tf.contrib.crf.crf_log_likelihood(y_pred,
                                                                             tf.cast(tf.keras.backend.argmax(y_true),
                                                                                     dtype=tf.int32),
                                                                             self.sequence_lengths,
                                                                             transition_params=self.transitions)
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        if self.mode == 'pad':
            data_shape = input_shape[0]
        else:
            data_shape = input_shape
        tf.TensorShape(data_shape).assert_has_rank(3)
        return data_shape[:2] + (self.output_dim,)

    @property
    def viterbi_accuracy(self):
        def accuracy(y_true, y_pred):
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            viterbi_sequence, _ = tf.contrib.crf.crf_decode(y_pred, self.transitions, sequence_lengths)
            output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
            return tf.keras.metrics.categorical_accuracy(y_true, output)

        accuracy.func_name = 'viterbi_accuracy'
        return accuracy
