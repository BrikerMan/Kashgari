# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py.py
@time: 2019-01-19 09:57

"""
from . import embedding
from .embedding import BERTEmbedding
from .embedding import CustomEmbedding
from .embedding import EmbeddingModel
from .embedding import Word2VecEmbedding
from .embedding import get_embedding_by_conf

if __name__ == "__main__":
    print("Hello world")
