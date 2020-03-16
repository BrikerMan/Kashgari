# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: utils.py
# time: 12:39 下午

import os
import json
import pydoc
import random
import numpy as np
from typing import List, Union, TypeVar, Tuple, Type

from tensorflow import keras

from kashgari import custom_objects
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.tasks.classification.abc_model import ABCClassificationModel
from kashgari.tasks.labeling.abc_model import ABCLabelingModel

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


def get_list_subset(target: List[T], index_list: List[int]) -> List[T]:
    """
    Get the subset of the target list
    Args:
        target: target list
        index_list: subset items index

    Returns:
        subset of the original list
    """
    return [target[i] for i in index_list if i < len(target)]


def unison_shuffled_copies(a: T1, b: T2) -> Tuple[T1, T2]:
    """
    Union shuffle two arrays
    Args:
        a:
        b:

    Returns:

    """
    data_type = type(a)
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    if data_type == np.ndarray:
        return np.array(a), np.array(b)
    return list(a), list(b)


def custom_object_scope():
    return keras.utils.custom_object_scope(custom_objects)


def load_model(model_path: str, load_weights: bool = True) -> Union[ABCClassificationModel, ABCLabelingModel]:
    """
    Load saved model from saved model from `model.save` function
    Args:
        model_path: model folder path
        load_weights: only load model structure and vocabulary when set to False, default True.

    Returns:
        Loaded kashgari model
    """
    with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    model_class: Type[ABCClassificationModel] = pydoc.locate(f"{model_info['module']}.{model_info['class_name']}")
    model_json_str = json.dumps(model_info['tf_model'])

    embed_info = model_info['embedding']
    embed_class: Type[ABCEmbedding] = pydoc.locate(f"{embed_info['module']}.{embed_info['class_name']}")
    embedding: ABCEmbedding = embed_class.load_saved_model_embedding(embed_info)

    model = model_class(embedding=embedding)
    model.tf_model = keras.models.model_from_json(model_json_str, custom_objects)
    if load_weights:
        model.tf_model.load_weights(os.path.join(model_path, 'model_weights.h5'))

    # Load Weights from model
    for layer in embedding.embed_model.layers:
        layer.set_weights(model.tf_model.get_layer(layer.name).get_weights())

    return model


if __name__ == "__main__":
    p = '/Users/brikerman/Desktop/python/Kashgari2/tests/test_classification/model'
    model = load_model(p)
    model.tf_model.summary()
