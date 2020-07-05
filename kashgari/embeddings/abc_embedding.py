# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abc_embedding.py
# time: 2:43 下午

import json
from typing import Dict, List, Any, Optional, Union

import numpy as np
import tensorflow as tf
import tqdm

import kashgari
from kashgari.generators import CorpusGenerator
from kashgari.logger import logger
from kashgari.processors import ABCProcessor

L = tf.keras.layers


class ABCEmbedding:
    def to_dict(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            'segment': self.segment,
            'embedding_size': self.embedding_size,
            'max_position': self.max_position,
            **self.kwargs
        }
        return {
            '__class_name__': self.__class__.__name__,
            '__module__': self.__class__.__module__,
            'config': config,
            'embed_model': json.loads(self.embed_model.to_json())
        }

    def __init__(self,
                 segment: bool = False,
                 embedding_size: int = 100,
                 max_position: int = None,
                 **kwargs: Any):

        self.embed_model: tf.keras.Model = None

        self.segment: bool = segment  # type: ignore
        self.kwargs = kwargs

        self.embedding_size: int = embedding_size  # type: ignore
        self.max_position: int = max_position  # type: ignore
        self.vocab2idx = self.load_embed_vocab()
        self._text_processor: Optional[ABCProcessor] = None

    def _override_load_model(self, config: Dict) -> None:
        embed_model_json_str = json.dumps(config['embed_model'])
        self.embed_model = tf.keras.models.model_from_json(embed_model_json_str,
                                                           custom_objects=kashgari.custom_objects)

    def setup_text_processor(self, processor: ABCProcessor) -> None:
        self._text_processor = processor
        self.build_embedding_model(vocab_size=processor.vocab_size)
        self._text_processor.segment = self.segment
        if self.vocab2idx:
            self._text_processor.vocab2idx = self.vocab2idx
            self._text_processor.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])

    def get_seq_length_from_corpus(self,
                                   generators: List[CorpusGenerator],
                                   *,
                                   use_label: bool = False,
                                   cover_rate: float = 0.95) -> int:
        """
        Calculate proper sequence length according to the corpus

        Args:
            generators:
            use_label:
            cover_rate:

        Returns:

        """
        seq_lens = []
        for gen in generators:
            for sentence, label in tqdm.tqdm(gen, desc="Calculating sequence length"):
                if use_label:
                    seq_lens.append(len(label))
                else:
                    seq_lens.append(len(sentence))
        if cover_rate == 1.0:
            target_index = -1
        else:
            target_index = int(cover_rate * len(seq_lens))
        sequence_length = sorted(seq_lens)[target_index]
        logger.debug(f'Calculated sequence length = {sequence_length}')
        return sequence_length

    def load_embed_vocab(self) -> Optional[Dict[str, int]]:
        """
        Load vocab dict from embedding layer

        Returns:
            vocab dict or None
        """
        raise NotImplementedError

    def build_embedding_model(self,
                              *,
                              vocab_size: int = None,
                              force: bool = False,
                              **kwargs: Dict) -> None:
        raise NotImplementedError

    def embed(self,
              sentences: List[List[str]],
              *,
              debug: bool = False) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentences: Sentence list to embed
            debug: show debug info
        Returns:
            vectorized sentence list
        """
        if self._text_processor is None:
            raise ValueError('Need to setup the `embedding.setup_text_processor` before calling the embed function.')

        tensor_x = self._text_processor.transform(sentences,
                                                  segment=self.segment,
                                                  seq_length=self.max_position)
        if debug:
            logger.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results


if __name__ == "__main__":
    pass
