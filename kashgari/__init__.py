# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py
@time: 2019-05-17 11:15

"""

import os
from distutils.version import LooseVersion
from typing import Any, Dict

os.environ["TF_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

custom_objects: Dict[str, Any] = {}


def check_tfa_version(tf_version: str) -> str:
    if LooseVersion(tf_version) < "2.2.0":
        return "0.9.1"
    elif LooseVersion(tf_version) < "2.3.0":
        return "0.11.2"
    else:
        return "0.13.0"


def dependency_check() -> None:
    import tensorflow as tf

    tfa_version = check_tfa_version(tf_version=tf.__version__)
    try:
        import tensorflow_addons as tfa
    except:
        raise ImportError(
            "Kashgari request tensorflow_addons, please install via the "
            f"`$pip install tensorflow_addons=={tfa_version}`"
        )


dependency_check()

from kashgari import corpus, embeddings, layers, macros, processors, tasks, utils
from kashgari.__version__ import __version__
from kashgari.macros import config

custom_objects = layers.resigter_custom_layers(custom_objects)
