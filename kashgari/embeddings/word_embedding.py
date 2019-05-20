# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: w2v_embedding.py
# time: 2019-05-20 17:32

import logging
from typing import Union, Optional, Dict, Any, List

import numpy as np
from gensim.models import KeyedVectors
from tensorflow import keras

from kashgari.embeddings.base_embedding import Embedding
from kashgari.pre_processors import PreProcessor

L = keras.layers


class WordEmbedding(Embedding):
    """Pre-trained word2vec embedding"""

    def __init__(self,
                 w2v_path: str,
                 w2v_kwargs: Dict[str, Any] = None,
                 sequence_length: Union[int, str] = 'auto',
                 processor: Optional[PreProcessor] = None,
                 **kwargs):
        """
        Args:
            w2v_path: word2vec file path
            w2v_kwargs: params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
        """
        super(WordEmbedding, self).__init__(sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor)

        if w2v_kwargs is None:
            w2v_kwargs = {}
        self.w2v_path = w2v_path
        self.w2v_kwargs = w2v_kwargs

    def _build_token2idx_from_w2v(self):
        w2v = KeyedVectors.load_word2vec_format(self.w2v_path, **self.w2v_kwargs)

        token2idx = {
            self.processor.token_pad: 0,
            self.processor.token_unk: 1,
            self.processor.token_bos: 2,
            self.processor.token_eos: 3
        }

        # 我们遍历预训练词嵌入的词表，加入到我们的标记索引词典
        for token in w2v.index2word:
            token2idx[token] = len(token2idx)

        # 初始化一个形状为 [标记总数，预训练向量维度] 的全 0 张量
        vector_matrix = np.zeros((len(token2idx), w2v.vector_size))
        # 随机初始化 <UNK> 标记的张量
        vector_matrix[1] = np.random.rand(w2v.vector_size)
        # 从索引 2 开始使用预训练的向量
        vector_matrix[4:] = w2v.vectors
        self.embedding_size = w2v.vector_size
        self.w2v_vector_matrix = vector_matrix
        self.w2v_token2idx = token2idx
        self.w2v_top_words = w2v.index2entity[:50]

        logging.debug('------------------------------------------------')
        logging.debug('Loaded gensim word2vec model')
        logging.debug('model        : {}'.format(self.w2v_path))
        logging.debug('word count   : {}'.format(len(self.w2v_vector_matrix)))
        logging.debug('Top 50 word  : {}'.format(self.w2v_top_words))
        logging.debug('------------------------------------------------')

    def build_model(self, **kwargs):
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            input_tensor = L.Input(shape=(self.sequence_length,),
                                   name='inputs')
            layer_embedding = L.Embedding(self.token_count,
                                          self.embedding_size,
                                          weights=[self.w2v_vector_matrix],
                                          trainable=False,
                                          name='layer_embedding')

            embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = keras.Model(input_tensor, embedded_tensor)

    def prepare_for_labeling(self,
                             x: List[List[str]],
                             y: List[List[str]]):
        if len(self.processor.token2idx) == 0:
            self._build_token2idx_from_w2v()

            self.processor.token2idx = self.w2v_token2idx
            self.processor.idx2token = dict([(v, k) for k, v in self.w2v_token2idx.items()])

        super(WordEmbedding, self).prepare_for_labeling(x, y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from kashgari.corpus import SMP2018ECDTCorpus

    test_x, test_y = SMP2018ECDTCorpus.load_data('valid')

    path = 'http://storage.eliyar.biz/embedding/word2vec/sample_w2v.txt'
    from tensorflow.python.keras.utils import get_file

    print(get_file(path))
    w = WordEmbedding(path, w2v_kwargs={'binary': True})
    w.prepare_for_labeling(test_x, test_y)
    w.embed_model.summary()
    print("Hello world")
