# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_embedding.py
# time: 2019-05-20 17:40

import logging
from typing import Union, List, Optional

import numpy as np
from tensorflow import keras
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from kashgari.pre_processors import PreProcessor

L = keras.layers


class Embedding(object):
    """Base class for Embedding Model"""

    def __init__(self,
                 sequence_length: Union[int, str] = 'auto',
                 embedding_size: int = 100,
                 processor: Optional[PreProcessor] = None):

        self.embedding_size = embedding_size
        self.sequence_length: Union[int, str] = sequence_length
        self.embed_model: Optional[keras.Model] = None

        if processor is None:
            self.processor = PreProcessor()
        else:
            self.processor = processor
            self.build_model()

    @property
    def token_count(self) -> int:
        return len(self.processor.token2idx)

    @property
    def sequence_length(self) -> Union[str, int]:
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, val: Union[str, int]):
        if isinstance(val, str):
            if val is 'auto':
                logging.warning("Sequence length will auto set at 95% of sequence length")
            elif val == 'variable':
                val = None
            else:
                raise ValueError("sequence_length must be an int or 'auto' or 'variable'")
        self._sequence_length = val

    def build_model(self, **kwargs):
        raise NotImplementedError

    def prepare_for_labeling(self,
                             x: List[List[str]],
                             y: List[List[str]]):
        """
        Prepare embedding layer and pre-processor for labeling task

        Args:
            x:
            y:

        Returns:

        """
        self.processor.prepare_labeling_dicts_if_need(x, y)
        if self.sequence_length == 'auto':
            self.sequence_length = self.processor.seq_length_95
        self.build_model()

    def embed(self, sentence: List[str]) -> np.array:
        return self.batch_embed([sentence])[0]

    def batch_embed(self, sentence_list: List[List[str]]) -> np.ndarray:
        numerized_token = [self.processor.numerize_token_sequence(sen) for sen in sentence_list]
        padded_token = pad_sequences(numerized_token, self.sequence_length, padding='post', truncating='post')
        embed_results = self.embed_model.predict(padded_token)
        return embed_results


if __name__ == "__main__":
    print("Hello world")
