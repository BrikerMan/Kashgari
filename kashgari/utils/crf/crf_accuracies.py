# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: crf_accuracies.py.py
@time: 2019-01-23 17:10

"""
from keras import backend as K

"""
from https://github.com/keras-team/keras-contrib
"""


def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
    y_pred = K.argmax(y_pred, -1)
    if sparse_target:
        y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
    else:
        y_true = K.argmax(y_true, -1)
    judge = K.cast(K.equal(y_pred, y_true), K.floatx())
    if mask is None:
        return K.mean(judge)
    else:
        mask = K.cast(mask, K.floatx())
        return K.sum(judge * mask) / K.sum(mask)


def crf_viterbi_accuracy(y_true, y_pred):
    '''Use Viterbi algorithm to get best path, and compute its accuracy.
    `y_pred` must be an output from CRF.'''
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.viterbi_decoding(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


def crf_marginal_accuracy(y_true, y_pred):
    '''Use time-wise marginal argmax as prediction.
    `y_pred` must be an output from CRF with `learn_mode="marginal"`.'''
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.get_marginal_prob(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


def crf_accuracy(y_true, y_pred):
    '''Ge default accuracy based on CRF `test_mode`.'''
    crf, idx = y_pred._keras_history[:2]
    if crf.test_mode == 'viterbi':
        return crf_viterbi_accuracy(y_true, y_pred)
    else:
        return crf_marginal_accuracy(y_true, y_pred)


if __name__ == "__main__":
    print("Hello world")
