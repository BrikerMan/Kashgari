# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: word2vec.py
@time: 2019-01-19 11:13

"""
import os
import json
import logging
import numpy as np
from typing import Union
from gensim.models import KeyedVectors
from kashgari import k


class Word2Vec(object):
    def __init__(self, model: Union[k.Word2VecModels, str], **kwargs):
        self.model = model
        self.model_path = k.get_model_path(model)
        self.keyed_vector: KeyedVectors = KeyedVectors.load_word2vec_format(self.model_path, **kwargs)
        self.embedding_size = self.keyed_vector.vector_size
        logging.debug('------------------------------------------------')
        logging.debug('Loaded gensim word2vec model')
        logging.debug('model        : {}'.format(self.model_path))
        logging.debug('word count   : {}'.format(len(self.keyed_vector.index2entity)))
        logging.debug('Top 50 word  : {}'.format(self.keyed_vector.index2entity[:50]))
        logging.debug('------------------------------------------------')

    def get_embedding_matrix(self) -> np.array:
        base_matrix = []

        file = os.path.join(k.DATA_PATH, 'w2v_embedding_{}.json'.format(self.embedding_size))
        if os.path.exists(file):
            base_matrix = json.load(open(file, 'r', encoding='utf-8'))
            base_matrix = [np.array(matrix) for matrix in base_matrix]
        else:
            for index, key in enumerate(k.MARKED_KEYS):
                if index != 0:
                    vector = np.random.uniform(-0.5, 0.5, self.embedding_size)
                else:
                    vector = np.zeros(self.embedding_size)
                base_matrix.append(vector)
            with open(file, 'w', encoding='utf-8') as f:
                f.write(json.dumps([list(item) for item in base_matrix]))

        matrix_list = base_matrix + list(self.keyed_vector.vectors)
        return np.array(matrix_list)


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    init_logger()
    w = Word2Vec(k.Word2VecModels.sgns_weibo_bigram, limit=100)
    logging.info(w.get_embedding_matrix()[0])
    logging.info(w.get_embedding_matrix().shape)
    logging.info("Hello world")
