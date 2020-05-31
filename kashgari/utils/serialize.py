# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: serialize.py
# time: 11:23 上午

import json

import tensorflow as tf
from json import JSONEncoder
from kashgari.processors import ABCProcessor
from kashgari.embeddings import ABCEmbedding


class KashgariEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, ABCProcessor):
            return o.to_dict()
        elif isinstance(o, ABCEmbedding):
            return o.to_dict()
        elif isinstance(o, tf.keras.Model):
            return json.loads(o.to_json())
        else:
            return super(KashgariEncoder, self).default(o)


if __name__ == "__main__":
    pass
