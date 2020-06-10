# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 11:22 上午

import warnings
import tensorflow as tf
from typing import TYPE_CHECKING, Union
from tensorflow.keras.utils import CustomObjectScope

from kashgari import custom_objects
from .data import get_list_subset
from .data import unison_shuffled_copies
from .multi_label import MultiLabelBinarizer
from .serialize import load_data_object

if TYPE_CHECKING:
    from kashgari.tasks.labeling import ABCLabelingModel
    from kashgari.tasks.classification import ABCClassificationModel


def custom_object_scope() -> CustomObjectScope:
    return tf.keras.utils.custom_object_scope(custom_objects)


def load_model(model_path: str) -> Union["ABCLabelingModel", "ABCClassificationModel"]:
    warnings.warn("The 'load_model' function is deprecated, "
                  "use 'XX_Model.load_model' instead", DeprecationWarning, 2)
    from kashgari.tasks.abs_task_model import ABCTaskModel
    return ABCTaskModel.load_model(model_path=model_path)


if __name__ == "__main__":
    pass
