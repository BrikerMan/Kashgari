# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bare_embedding.py
# time: 2019-05-20 10:36
import logging
from enum import Enum, unique
from typing import Union, List, Optional

from tensorflow import keras

from kashgari.pre_processors import PreProcessor

L = keras.layers


@unique
class TaskType(Enum):
    labeling = 'labeling'
    classification = 'classification'


class BareEmbedding(object):

    """Embedding layer without pre-training, train embedding layer while training model"""

    def __init__(self,
                 sequence_length: Union[int, str] = 'auto',
                 embedding_size: int = 100,
                 processor: Optional[PreProcessor] = None):
        """
        Init bare embedding (embedding without pre-training)

        Args:
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            embedding_size: Dimension of the dense embedding.
        """

        self.embedding_size = embedding_size
        self.sequence_length: Union[int, str] = sequence_length
        self.embed_model: Optional[keras.Model] = None

        if processor is None:
            self.processor = PreProcessor()
        else:
            self.processor = processor
            self.build_model()

    @property
    def token_count(self):
        return len(self.processor.token2idx)

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, val):
        if isinstance(val, str):
            if val is 'auto':
                logging.warning("Sequence length will auto set at 95% of sequence length")
            elif val == 'variable':
                val = None
            else:
                raise ValueError("sequence_length must be an int or 'auto' or 'variable'")
        self._sequence_length = val

    def build_model(self, **kwargs):
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            if self.sequence_length == 'auto':
                self.sequence_length = self.processor.seq_length_95

            input_tensor = L.Input(shape=(self.sequence_length,),
                                   name='inputs')
            layer_embedding = L.Embedding(self.token_count,
                                          self.embedding_size,
                                          name='layer_embedding')

            embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = keras.Model(input_tensor, embedded_tensor)

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
        self.build_model()


if __name__ == "__main__":
    pass
