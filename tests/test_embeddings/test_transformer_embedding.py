# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_transformer_embedding.py
# time: 2:47 下午

from tensorflow.keras.utils import get_file

from kashgari.embeddings import BertEmbedding
from kashgari.macros import DATA_PATH
from tests.test_embeddings.test_bare_embedding import TestBareEmbedding


class TestTransferEmbedding(TestBareEmbedding):

    def build_embedding(self):
        bert_path = get_file('bert_sample_model',
                             "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                             cache_dir=DATA_PATH,
                             untar=True)
        embedding = BertEmbedding(model_folder=bert_path)
        return embedding


if __name__ == "__main__":
    pass
