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

import kashgari
from kashgari.processors import ClassificationProcessor, LabelingProcessor
from kashgari.processors.base_processor import BaseProcessor

L = keras.layers


class Embedding(object):
    """Base class for Embedding Model"""

    def info(self):
        return {
            'sequence_length': self.sequence_length,
            'processor': self.processor.info()
        }

    def __init__(self,
                 task: str = None,
                 sequence_length: Union[int, str] = 'auto',
                 embedding_size: int = 100,
                 processor: Optional[BaseProcessor] = None):

        self.embedding_size = embedding_size
        self.sequence_length: Union[int, str] = sequence_length
        self.embed_model: Optional[keras.Model] = None

        if processor is None:
            if task == kashgari.CLASSIFICATION:
                self.processor = ClassificationProcessor()
            elif task == kashgari.LABELING:
                self.processor = LabelingProcessor()
            else:
                raise ValueError()
        else:
            self.processor = processor

    @property
    def token_count(self) -> int:
        """
        corpus token count
        """
        return len(self.processor.token2idx)

    @property
    def sequence_length(self) -> Union[int, str]:
        """
        model sequence length
        """
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, val: Union[int, str]):
        if isinstance(val, str):
            if val == 'auto':
                logging.warning("Sequence length will auto set at 95% of sequence length")
            elif val == 'variable':
                val = None
            else:
                raise ValueError("sequence_length must be an int or 'auto' or 'variable'")
        self._sequence_length = val

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def analyze_corpus(self,
                       x: List[List[str]],
                       y: Union[List[List[str]], List[str]]):
        """
        Prepare embedding layer and pre-processor for labeling task

        Args:
            x:
            y:

        Returns:

        """
        self.processor.analyze_corpus(x, y)
        if self.sequence_length == 'auto':
            self.sequence_length = self.processor.dataset_info['RECOMMEND_LEN']
        self._build_model()

    def embed_one(self, sentence: Union[List[str], List[int]]) -> np.array:
        """
        Convert one sentence to vector

        Args:
            sentence: target sentence, list of str

        Returns:
            vectorized sentence
        """
        return self.embed([sentence])[0]

    def embed(self,
              sentence_list: Union[List[List[str]], List[List[int]]],
              debug: bool = False) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentence_list: Sentence list to embed
            debug: show debug info
        Returns:
            vectorized sentence list
        """
        tensor_x = self.process_x_dataset(sentence_list)

        if debug:
            logging.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results

    def process_x_dataset(self,
                          data: List[List[str]],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        """
        batch process feature data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        return self.processor.process_x_dataset(data, self.sequence_length, subset)

    def process_y_dataset(self,
                          data: List[List[str]],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        """
        batch process labels data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        return self.processor.process_y_dataset(data, self.sequence_length, subset)

    def reverse_numerize_label_sequences(self,
                                         sequences,
                                         lengths=None):
        return self.processor.reverse_numerize_label_sequences(sequences, lengths=lengths)

    def __repr__(self):
        return f"<{self.__class__} seq_len: {self.sequence_length}>"

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    print("Hello world")
