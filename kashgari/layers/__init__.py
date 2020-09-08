# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 7:39 下午

from typing import Dict, Any
from tensorflow import keras

from .conditional_random_field import KConditionalRandomField
from .behdanau_attention import BahdanauAttention  # type: ignore

L = keras.layers
L.BahdanauAttention = BahdanauAttention
L.KConditionalRandomField = KConditionalRandomField


def resigter_custom_layers(custom_objects: Dict[str, Any]) -> Dict[str, Any]:
    custom_objects['KConditionalRandomField'] = KConditionalRandomField
    custom_objects['BahdanauAttention'] = BahdanauAttention
    return custom_objects


if __name__ == "__main__":
    pass
