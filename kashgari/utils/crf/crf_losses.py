# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: crf_losss.py
@time: 2019-01-23 17:08

"""
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy

"""
from https://github.com/keras-team/keras-contrib
"""


def crf_nll(y_true, y_pred):
    """The negative log-likelihood for linear chain Conditional Random Field (CRF).

    This loss function is only used when the `layers.CRF` layer
    is trained in the "join" mode.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        A scalar representing corresponding to the negative log-likelihood.

    # Raises
        TypeError: If CRF is not the last layer.

    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.
    """

    crf, idx = y_pred._keras_history[:2]
    if crf._outbound_nodes:
        raise TypeError('When learn_model="join", CRF must be the last layer.')
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):
    """General CRF loss function depending on the learning mode.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        If the CRF layer is being trained in the join mode, returns the negative
        log-likelihood. Otherwise returns the categorical crossentropy implemented
        by the underlying Keras backend.

    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)