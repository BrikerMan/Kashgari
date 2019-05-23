# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: stacked_embedding.py
# time: 2019-05-23 09:18

from typing import Union, Optional, Tuple, List

import numpy as np
import tensorflow as tf

import kashgari
from kashgari.embeddings.base_embedding import Embedding
from kashgari.processors.base_processor import BaseProcessor
from kashgari.layers import L


# Todo: A better name for this class
class StackedEmbedding(Embedding):
    """Embedding layer without pre-training, train embedding layer while training model"""

    def __init__(self,
                 embeddings: List[Embedding],
                 processor: Optional[BaseProcessor] = None):
        """
        Init bare embedding (embedding without pre-training)

        Args:
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            embedding_size: Dimension of the dense embedding.
        """
        task = kashgari.CLASSIFICATION
        if all(isinstance(embed.sequence_length, int) for embed in embeddings):
            sequence_length = [embed.sequence_length for embed in embeddings]
        else:
            raise ValueError('Need to set sequence length for all embeddings while using stacked embedding')

        super(StackedEmbedding, self).__init__(task=task,
                                               sequence_length=sequence_length[0],
                                               embedding_size=100,
                                               processor=processor)
        self.embeddings = embeddings
        self.processor = embeddings[0].processor
        self._build_model()

    def _build_model(self, **kwargs):
        if self.embed_model is None and all(embed.embed_model is not None for embed in self.embeddings):
            layer_concatenate = L.Concatenate(name='layer_concatenate')
            inputs = [embed.embed_model.input for embed in self.embeddings]
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
            result.append(self.embeddings[index].process_x_dataset(dataset, subset))
        return tuple(result)

    def process_y_dataset(self,
                          data: List[List[str]],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        return self.embeddings[0].process_y_dataset(data, subset)


if __name__ == "__main__":
    pass
