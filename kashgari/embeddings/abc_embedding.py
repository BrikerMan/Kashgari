# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abc_embedding.py
# time: 2:43 下午

import logging
import numpy as np
from typing import Dict, Any, List
from tensorflow import keras

from kashgari.typing import TextSamplesVar, LabelSamplesVar
from kashgari.processor.abc_processor import ABCProcessor
from kashgari.processor import SequenceProcessor
from kashgari.generators import CorpusGenerator

L = keras.layers


class ABCEmbedding:
    def __init__(self,
                 sequence_length: int = None,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None,
                 **kwargs):

        if text_processor is None:
            text_processor = SequenceProcessor()

        self.text_processor = text_processor
        self.label_processor = label_processor
        self.embed_model: keras.Model = None

        if sequence_length is not None:
            self.text_processor.sequence_length = sequence_length

    def set_sequence_length(self, length: int):
        self.text_processor.sequence_length = length
        if self.embed_model is not None:
            logging.info(f"Rebuild embedding model with sequence length: {length}")
            self.embed_model = None
            self.build_embedding_model()

    def build(self, x_data: TextSamplesVar, y_data: LabelSamplesVar):
        gen = CorpusGenerator(x_data=x_data, y_data=y_data)
        self.build_generator(gen)

    def build_generator(self, gen: CorpusGenerator = None):
        self.build_text_vocab(gen=gen)
        self.build_embedding_model()
        if self.label_processor and gen is not None:
            self.label_processor.build_vocab_dict_if_needs(gen)

    def build_text_vocab(self, gen: CorpusGenerator, force=False):
        raise NotImplementedError

    def build_embedding_model(self):
        raise NotImplementedError

    def embed(self,
              sentences: List[List[str]],
              debug: bool = False) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentences: Sentence list to embed
            debug: show debug info
        Returns:
            vectorized sentence list
        """
        tensor_x = self.text_processor.numerize_samples(sentences)

        if debug:
            logging.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results


if __name__ == "__main__":
    pass
