# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: loss.py
# time: 2019-05-22 16:09

from tensorflow.python.keras import backend


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = backend.variable(weights)

    def categorical_crossentropy(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
        # calc
        loss = y_true * backend.log(y_pred) * weights
        loss = -backend.sum(loss, -1)
        return loss

    return categorical_crossentropy


if __name__ == "__main__":
    print("Hello world")
