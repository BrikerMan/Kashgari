# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: helpers.py
@time: 2019-05-17 11:37

"""
import os
import json
import pickle
import random
import pathlib
import keras_bert
from pydoc import locate
import tensorflow as tf
from kashgari.tasks.base_model import BaseModel
from typing import List, Tuple, Optional, Any


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def get_list_subset(target: List, index_list: List[int]) -> List:
    return [target[i] for i in index_list if i < len(target)]


def get_tuple_item(data: Optional[Tuple], index: int) -> Optional[Any]:
    if data and len(data) > index:
        return data[index]
    else:
        return None


def get_project_path() -> str:
    here = pathlib.Path(__file__).parent
    return os.path.abspath(os.path.join(here, '../'))


def wrap_as_tuple(original) -> Tuple[Any]:
    if isinstance(original, tuple):
        return original
    elif isinstance(original, list) or len(original) == 1:
        return tuple([original])
    return original


def get_custom_objects():
    return keras_bert.get_custom_objects()


def custom_object_scope():
    return tf.keras.utils.custom_object_scope(get_custom_objects())


def load_model(model_path: str) -> BaseModel:
    with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
        config = json.load(f)

    processor_path = os.path.join(model_path, 'processor.pickle')
    processor = pickle.load(open(processor_path, "rb"))

    from kashgari.embeddings import BareEmbedding
    embedding = BareEmbedding(processor=processor,
                              sequence_length=tuple(config['embedding']['sequence_length']))

    task_module = locate(f"{config['module']}.{config['architect_name']}")
    model: BaseModel = task_module(embedding=embedding, hyper_parameters=config['hyper_parameters'])
    model.tf_model = tf.keras.models.load_model(os.path.join(model_path, 'model.h5'),
                                                custom_objects=get_custom_objects())
    return model


if __name__ == "__main__":
    print(get_project_path())
