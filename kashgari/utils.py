# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: utils.py
# time: 12:39 下午

import json
import os
import pydoc
import random
from typing import List, Union, TypeVar, Tuple, Type, Dict, Any

import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope

from kashgari import custom_objects
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.tasks.classification.abc_model import ABCClassificationModel
from kashgari.tasks.labeling.abc_model import ABCLabelingModel
from kashgari.types import MultiLabelClassificationLabelVar


def custom_object_scope() -> CustomObjectScope:
    return keras.utils.custom_object_scope(custom_objects)


def load_object(data: Dict, **kwargs: Dict) -> Any:
    """
    Load Object From Dict
    Args:
        data:
        **kwargs:

    Returns:

    """
    module_name = f"{data['__module__']}.{data['__class_name__']}"
    obj = pydoc.locate(module_name)(**data['config'], **kwargs)  # type: ignore

    return obj


ModelTypeVar = Union[ABCClassificationModel, ABCLabelingModel]


def load_model(model_path: str, load_weights: bool = True) -> ModelTypeVar:
    """
    Load saved model from saved model from `model.save` function
    Args:
        model_path: model folder path
        load_weights: only load model structure and vocabulary when set to False, default True.

    Returns:
        Loaded kashgari model
    """
    # from bert4keras.layers import ConditionalRandomField
    # with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
    #     model_info = json.load(f)

    # print(model_info.keys())
    # text_processor = _load_object_from_json(model_info['text_processor'])
    # label_processor = _load_object_from_json(model_info['label_processor'])
    #
    # embed_info = model_info['embedding']
    # embed_class: Type[ABCEmbedding] = pydoc.locate(  # type: ignore
    #     f"{embed_info['module']}.{embed_info['class_name']}")
    # embedding: ABCEmbedding = embed_class.load_saved_model_embedding(embed_info)
    #
    # model: ModelTypeVar = _load_object_from_json(model_info,
    #                                              embedding=embedding,
    #                                              text_processor=text_processor,
    #                                              label_processor=label_processor)
    #
    # model_json_str = json.dumps(model_info['tf_model'])
    # model.tf_model = keras.models.model_from_json(model_json_str, custom_objects)
    # if load_weights:
    #     model.tf_model.load_weights(os.path.join(model_path, 'model_weights.h5'))
    #
    # # Load Weights from model
    # for layer in embedding.embed_model.layers:
    #     layer.set_weights(model.tf_model.get_layer(layer.name).get_weights())
    #
    # if isinstance(model.tf_model.layers[-1], ConditionalRandomField):
    #     model.layer_crf = model.tf_model.layers[-1]  # type: ignore

    return None



if __name__ == "__main__":
    pass
