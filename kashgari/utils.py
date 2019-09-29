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
import json
import os
import pathlib
import pydoc
import random
import time
from typing import List, Optional, Dict, Union

import tensorflow as tf
from tensorflow.python import keras, saved_model

from kashgari import custom_objects
from kashgari.embeddings.base_embedding import Embedding
from kashgari.layers.crf import CRF
from kashgari.processors.base_processor import BaseProcessor
from kashgari.tasks.base_model import BaseModel
from kashgari.tasks.classification.base_model import BaseClassificationModel
from kashgari.tasks.labeling.base_model import BaseLabelingModel


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def get_list_subset(target: List, index_list: List[int]) -> List:
    return [target[i] for i in index_list if i < len(target)]


def custom_object_scope():
    return tf.keras.utils.custom_object_scope(custom_objects)


def load_model(model_path: str, load_weights: bool = True) -> Union[BaseClassificationModel, BaseLabelingModel]:
    """
    Load saved model from saved model from `model.save` function
    Args:
        model_path: model folder path
        load_weights: only load model structure and vocabulary when set to False, default True.

    Returns:

    """
    with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    model_class = pydoc.locate(f"{model_info['module']}.{model_info['class_name']}")
    model_json_str = json.dumps(model_info['tf_model'])

    model = model_class()
    model.tf_model = tf.keras.models.model_from_json(model_json_str, custom_objects)
    if load_weights:
        model.tf_model.load_weights(os.path.join(model_path, 'model_weights.h5'))

    embed_info = model_info['embedding']
    embed_class = pydoc.locate(f"{embed_info['module']}.{embed_info['class_name']}")
    embedding: Embedding = embed_class._load_saved_instance(embed_info,
                                                            model_path,
                                                            model.tf_model)

    model.embedding = embedding

    if type(model.tf_model.layers[-1]) == CRF:
        model.layer_crf = model.tf_model.layers[-1]

    return model


def load_processor(model_path: str) -> BaseProcessor:
    """
    Load processor from model
    When we using tf-serving, we need to use model's processor to pre-process data
    Args:
        model_path:

    Returns:

    """
    with open(os.path.join(model_path, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    processor_info = model_info['embedding']['processor']
    processor_class = pydoc.locate(f"{processor_info['module']}.{processor_info['class_name']}")
    processor: BaseProcessor = processor_class(**processor_info['config'])
    return processor


def convert_to_saved_model(model: BaseModel,
                           model_path: str,
                           version: str = None,
                           inputs: Optional[Dict] = None,
                           outputs: Optional[Dict] = None):
    """
    Export model for tensorflow serving
    Args:
        model: Target model
        model_path: The path to which the SavedModel will be stored.
        version: The model version code, default timestamp
        inputs: dict mapping string input names to tensors. These are added
            to the SignatureDef as the inputs.
        outputs:  dict mapping string output names to tensors. These are added
            to the SignatureDef as the outputs.
    """
    pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
    if version is None:
        version = round(time.time())
    export_path = os.path.join(model_path, str(version))

    if inputs is None:
        inputs = {i.name: i for i in model.tf_model.inputs}
    if outputs is None:
        outputs = {o.name: o for o in model.tf_model.outputs}
    sess = keras.backend.get_session()
    saved_model.simple_save(session=sess,
                            export_dir=export_path,
                            inputs=inputs,
                            outputs=outputs)

    with open(os.path.join(export_path, 'model_info.json'), 'w') as f:
        f.write(json.dumps(model.info(), indent=2, ensure_ascii=True))
        f.close()


if __name__ == "__main__":
    path = '/Users/brikerman/Desktop/python/Kashgari/tests/classification/saved_models/' \
           'kashgari.tasks.classification.models/BiLSTM_Model'
    p = load_processor(path)
    print(p.process_x_dataset([list('语言模型')]))
    print(p.label2idx)
    print(p.token2idx)
