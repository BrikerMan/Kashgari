# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: word_embedding.py
# time: 3:06 下午

import logging
import numpy as np
from typing import Dict, Any, List
from gensim.models import KeyedVectors
from kashgari.processor.abs_processor import ABCProcessor
from kashgari.processor import SequenceProcessor
from tensorflow import keras
from kashgari.generators import CorpusGenerator

L = keras.layers


class WordEmbedding:
    def __init__(self,
                 w2v_path: str,
                 w2v_kwargs: Dict[str, Any] = None,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None):
        if w2v_kwargs is None:
            w2v_kwargs = {}

        if text_processor is None:
            text_processor = SequenceProcessor()

        self.w2v_path = w2v_path
        self.w2v_kwargs = w2v_kwargs

        self.text_processor = text_processor
        self.label_processor = label_processor
        self.embed_model: keras.Model = None

        self.embedding_size = None
        self.w2v_matrix = None

    def set_sequence_length(self, length: int):
        self.text_processor.sequence_length = length
        del self.embed_model
        self.build_embedding_model()

    def build(self, gen: CorpusGenerator = None):
        self.build_text_vocab(gen=gen)
        self.build_embedding_model()
        if self.label_processor and gen:
            self.label_processor.build_vocab_dict_if_needs(gen)

    def build_text_vocab(self, gen: CorpusGenerator, force=False):
        if force or self.w2v_matrix is None:
            w2v = KeyedVectors.load_word2vec_format(self.w2v_path, **self.w2v_kwargs)

            token2idx = {
                self.text_processor.token_pad: 0,
                self.text_processor.token_unk: 1,
                self.text_processor.token_bos: 2,
                self.text_processor.token_eos: 3
            }

            for token in w2v.index2word:
                token2idx[token] = len(token2idx)

            vector_matrix = np.zeros((len(token2idx), w2v.vector_size))
            vector_matrix[1] = np.random.rand(w2v.vector_size)
            vector_matrix[4:] = w2v.vectors

            self.text_processor.vocab2idx = token2idx
            self.text_processor.idx2vocab = dict([(v, k) for k, v in token2idx.items()])
            if gen:
                self.text_processor.build_vocab_dict_if_needs(generator=gen)
            self.embedding_size = w2v.vector_size
            self.w2v_matrix = vector_matrix
            w2v_top_words = w2v.index2entity[:50]

            logging.debug('------------------------------------------------')
            logging.debug('Loaded gensim word2vec model')
            logging.debug('model        : {}'.format(self.w2v_path))
            logging.debug('word count   : {}'.format(len(self.w2v_matrix)))
            logging.debug('Top 50 word  : {}'.format(w2v_top_words))
            logging.debug('------------------------------------------------')
        else:
            logging.debug("Already built the vocab")

    def build_embedding_model(self):
        if self.embed_model is None:
            input_tensor = L.Input(shape=(self.text_processor.sequence_length,),
                                   name=f'input')
            layer_embedding = L.Embedding(self.text_processor.vocab_size,
                                          self.embedding_size,
                                          weights=[self.w2v_matrix],
                                          trainable=False,
                                          name=f'layer_embedding')

            embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = keras.Model(input_tensor, embedded_tensor)

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
