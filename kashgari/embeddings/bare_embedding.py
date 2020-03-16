# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bare_embedding.py
# time: 2:17 下午

from tensorflow import keras

from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor

L = keras.layers


class BareEmbedding(ABCEmbedding):
    def __init__(self,
                 sequence_length: int = None,
                 embedding_size: int = 100,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None,
                 **kwargs):
        super(BareEmbedding, self).__init__(sequence_length=sequence_length,
                                            text_processor=text_processor,
                                            label_processor=label_processor,
                                            **kwargs)
        self.embedding_size = embedding_size

    def build_text_vocab(self, gen: CorpusGenerator = None, force=False):
        if force or not self.text_processor.is_vocab_build:
            self.text_processor.build_vocab_dict_if_needs(generator=gen)

    def build_embedding_model(self):
        if self.embed_model is None:
            input_tensor = L.Input(shape=(None,),
                                   name=f'input')
            layer_embedding = L.Embedding(self.text_processor.vocab_size,
                                          self.embedding_size,
                                          name=f'layer_embedding')

            embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = keras.Model(input_tensor, embedded_tensor)


if __name__ == "__main__":
    pass
