#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/2 9:19 下午
# File    : conditional_random_field.py
# Project : Kashgari

# mypy: ignore-errors

from distutils.version import LooseVersion

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


class KConditionalRandomField(tf.keras.layers.Layer):
    """
    K is to mark Kashgari version of CRF
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).

    Args:
        num_labels (int): the number of labels to tag each temporal input.

    Input shape:
        nD tensor with shape `(batch_size, sentence length, num_classes)`.

    Output shape:
        nD tensor with shape: `(batch_size, sentence length, num_classes)`.

    Masking
        This layer supports keras masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an embedding layer with the `mask_zero` parameter
        set to `True` or add a Masking Layer before this Layer
    """

    def __init__(self,
                 sparse_target=True,
                 **kwargs):
        if LooseVersion(tf.__version__) < '2.2.0':
            raise ImportError("The KConditionalRandomField requires TensorFlow 2.2.x version or higher.")

        super().__init__()
        self.transitions = kwargs.pop('transitions', None)
        self.output_dim = kwargs.pop('output_dim', None)
        self.sparse_target = sparse_target
        self.sequence_lengths = None
        self.mask = None

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "transitions": K.eval(self.transitions),
        }
        base_config = super().get_config()
        return dict(**base_config, **config)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        assert len(input_shape) == 3
        self.transitions = self.add_weight(
            name="transitions",
            shape=[input_shape[-1], input_shape[-1]],
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            self.sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
            self.mask = mask
        else:
            self.sequence_lengths = K.sum(K.ones_like(inputs[:, :, 0], dtype='int32'), axis=-1)
        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, self.sequence_lengths
        )
        output = K.cast(K.one_hot(viterbi_sequence, inputs.shape[-1]), inputs.dtype)
        return K.in_train_phase(inputs, output)

    def loss(self, y_true, y_pred):
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)
        log_likelihood, self.transitions = tfa.text.crf_log_likelihood(
            y_pred,
            y_true,
            self.sequence_lengths,
            transition_params=self.transitions,
        )
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.out_dim,)

    # use crf decode to estimate accuracy
    def accuracy(self, y_true, y_pred):
        mask = self.mask
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)

        y_pred, _ = tfa.text.crf_decode(
            y_pred, self.transitions, self.sequence_lengths
        )
        y_true = K.cast(y_true, y_pred.dtype)
        is_equal = K.equal(y_true, y_pred)
        is_equal = K.cast(is_equal, y_pred.dtype)
        if mask is None:
            return K.mean(is_equal)
        else:
            mask = K.cast(mask, y_pred.dtype)
            return K.sum(is_equal * mask) / K.sum(mask)

    # Use argmax to estimate accuracy
    def fast_accuracy(self, y_true, y_pred):
        mask = self.mask
        if len(K.int_shape(y_true)) == 3:
            y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, -1)
        y_true = K.cast(y_true, y_pred.dtype)
        # 逐标签取最大来粗略评测训练效果
        isequal = K.equal(y_true, y_pred)
        isequal = K.cast(isequal, y_pred.dtype)
        if mask is None:
            return K.mean(isequal)
        else:
            mask = K.cast(mask, y_pred.dtype)
            return K.sum(isequal * mask) / K.sum(mask)
