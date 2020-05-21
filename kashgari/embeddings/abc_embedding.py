# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abc_embedding.py
# time: 2:43 下午

import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import tqdm
from tensorflow import keras

import kashgari
from kashgari.generators import CorpusGenerator
from kashgari.processors import ABCProcessor

L = keras.layers


class ABCEmbedding:
    def info(self) -> Dict:
        config: Dict[str, Any] = {
            'segment': self.segment,
            'embedding_size': self.embedding_size,
            'max_position': self.max_position,
            **self.kwargs
        }
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'config': config,
            'embed_model': json.loads(self.embed_model.to_json())
        }

    @classmethod
    def load_saved_model_embedding(cls,
                                   config_dict: Dict,
                                   **kwargs: Any) -> 'ABCEmbedding':

        instance = cls(**config_dict['config'])

        embed_model_json_str = json.dumps(config_dict['embed_model'])
        instance.embed_model = keras.models.model_from_json(embed_model_json_str,
                                                            custom_objects=kashgari.custom_objects)
        return instance

    def __init__(self,
                 segment: bool = False,
                 embedding_size: int = 100,
                 max_position: int = None,
                 **kwargs: Any):

        self.embed_model: keras.Model = None

        self.segment: bool = segment  # type: ignore
        self.kwargs = kwargs

        self.embedding_size: int = embedding_size  # type: ignore
        self.max_position: int = max_position  # type: ignore
        self._text_processor: Optional[ABCProcessor] = None
        self._embedding_vocab2idx = self.load_embed_vocab()

    @property
    def embedding_vocab2idx(self) -> Dict[str, int]:
        return self._embedding_vocab2idx

    def setup_text_processor(self, processor: ABCProcessor) -> None:
        self._text_processor = processor
        self.build_embedding_model(vocab_size=processor.vocab_size)
        vocab2idx = self.load_embed_vocab()
        if vocab2idx:
            self._text_processor.vocab2idx = vocab2idx
            self._text_processor.idx2vocab = dict([(v, k) for k, v in vocab2idx.items()])

    def get_seq_length_from_corpus(self,
                                   corpus_gen: CorpusGenerator,
                                   *,
                                   use_label: bool = False,
                                   cover_rate: float = 0.95) -> int:
        """
        Calculate proper sequence length according to the corpus

        Args:
            corpus_gen:
            use_label:
            cover_rate:

        Returns:

        """
        seq_lens = []
        for sentence, label in tqdm.tqdm(corpus_gen, desc="Calculating sequence length"):
            if use_label:
                seq_lens.append(len(label))
            else:
                seq_lens.append(len(sentence))
        if cover_rate == 1.0:
            target_index = -1
        else:
            target_index = int(cover_rate * len(seq_lens))
        sequence_length = sorted(seq_lens)[target_index]
        logging.debug(f'Calculated sequence length = {sequence_length}')
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
        tensor_x = self._text_processor.transform(sentences, segment=self.segment)
        print(self.segment)
        print(tensor_x)
        if debug:
            logging.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results


if __name__ == "__main__":
    pass
