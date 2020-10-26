# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: tools.py
# time: 11:24 上午

import json
import os
from typing import Tuple

from kashgari.processors.abc_processor import ABCProcessor
from kashgari.utils.serialize import load_data_object


def load_processors_from_model(model_path: str) -> Tuple[ABCProcessor, ABCProcessor]:
    with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
        model_config = json.loads(f.read())
        text_processor: ABCProcessor = load_data_object(model_config['text_processor'])
        label_processor: ABCProcessor = load_data_object(model_config['label_processor'])

        sequence_length_from_saved_model = model_config['config'].get('sequence_length', None)
        text_processor._sequence_length_from_saved_model = sequence_length_from_saved_model
        label_processor._sequence_length_from_saved_model = sequence_length_from_saved_model

        return text_processor, label_processor


if __name__ == "__main__":
    text_processor, label_processor = load_processors_from_model('/Users/brikerman/Desktop/tf-serving/1603683152')
    x = text_processor.transform([list('我想你了')])
    print(x.tolist())
