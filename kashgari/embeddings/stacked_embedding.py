# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: stacked_embedding.py
# time: 2019-05-23 09:18

import json
import pydoc
from typing import Union, Optional, Tuple, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

import kashgari
from kashgari.embeddings.base_embedding import Embedding
from kashgari.layers import L
from kashgari.processors.base_processor import BaseProcessor


class StackedEmbedding(Embedding):
    """Embedding layer without pre-training, train embedding layer while training model"""

    @classmethod
    def _load_saved_instance(cls,
                             config_dict: Dict,
                             model_path: str,
                             tf_model: keras.Model):
        embeddings = []
        for embed_info in config_dict['embeddings']:
            embed_class = pydoc.locate(f"{embed_info['module']}.{embed_info['class_name']}")
            embedding: Embedding = embed_class._load_saved_instance(embed_info,
                                                                    model_path,
                                                                    tf_model)
            embeddings.append(embedding)
        instance = cls(embeddings=embeddings,
                       from_saved_model=True)
        print('----')
        print(instance.embeddings)

        embed_model_json_str = json.dumps(config_dict['embed_model'])
        instance.embed_model = keras.models.model_from_json(embed_model_json_str,
                                                            custom_objects=kashgari.custom_objects)
        # Load Weights from model
        for layer in instance.embed_model.layers:
            layer.set_weights(tf_model.get_layer(layer.name).get_weights())
        return instance

    def info(self):
        info = super(StackedEmbedding, self).info()
        info['embeddings'] = [embed.info() for embed in self.embeddings]
        info['config'] = {}
        return info

    def __init__(self,
                 embeddings: List[Embedding],
                 processor: Optional[BaseProcessor] = None,
                 from_saved_model: bool = False):
        """

        Args:
            embeddings:
            processor:
        """
        task = kashgari.CLASSIFICATION
        if all(isinstance(embed.sequence_length, int) for embed in embeddings):
            sequence_length = [embed.sequence_length for embed in embeddings]
        else:
            raise ValueError('Need to set sequence length for all embeddings while using stacked embedding')

        super(StackedEmbedding, self).__init__(task=task,
                                               sequence_length=sequence_length[0],
                                               embedding_size=100,
                                               processor=processor,
                                               from_saved_model=from_saved_model)

        self.embeddings = embeddings
        self.processor = embeddings[0].processor

        if not from_saved_model:
            self._build_model()

    def _build_model(self, **kwargs):
        if self.embed_model is None and all(embed.embed_model is not None for embed in self.embeddings):
            layer_concatenate = L.Concatenate(name='layer_concatenate')

            inputs = []

            for embed in self.embeddings:
                inputs += embed.embed_model.inputs

            # inputs = [embed.embed_model.inputs for embed in self.embeddings]
            outputs = layer_concatenate([embed.embed_model.output for embed in self.embeddings])

            self.embed_model = tf.keras.Model(inputs, outputs)

    def analyze_corpus(self,
                       x: Union[Tuple[List[List[str]], ...], List[List[str]]],
                       y: Union[List[List[str]], List[str]]):
        for index in range(len(x)):
            self.embeddings[index].analyze_corpus(x[index], y)
        self._build_model()

    def process_x_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        """
        batch process feature data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        result = []
        for index, dataset in enumerate(data):
            x = self.embeddings[index].process_x_dataset(dataset, subset)
            if isinstance(x, tuple):
                result += list(x)
            else:
                result.append(x)
        return tuple(result)

    def process_y_dataset(self,
                          data: List[List[str]],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        return self.embeddings[0].process_y_dataset(data, subset)


if __name__ == "__main__":
    pass
