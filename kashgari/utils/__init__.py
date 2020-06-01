# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 11:22 上午

from .data import get_list_subset
from .data import unison_shuffled_copies
from .serialize import load_data_object
from .multi_label import MultiLabelBinarizer

import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

from kashgari import custom_objects


def custom_object_scope() -> CustomObjectScope:
    return tf.keras.utils.custom_object_scope(custom_objects)


if __name__ == "__main__":
    pass
