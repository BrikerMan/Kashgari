# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bare_embedding.py
# time: 2019-05-20 10:36
import logging
from typing import Union, Optional

from tensorflow import keras

from kashgari.embeddings.base_embedding import Embedding
from kashgari.pre_processors import PreProcessor

L = keras.layers


# Todo: A better name for this class
class BareEmbedding(Embedding):

    """Embedding layer without pre-training, train embedding layer while training model"""

    def __init__(self,
                 sequence_length: Union[Tuple[int], str] = 'auto',
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
        super(BareEmbedding, self).__init__(sequence_length=sequence_length,
                                            embedding_size=embedding_size,
                                            processor=processor)

    def build_model(self, **kwargs):
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            input_tensor = L.Input(shape=(self.sequence_length,),
                                   name='inputs')
            layer_embedding = L.Embedding(self.token_count,
                                          self.embedding_size,
                                          name='layer_embedding')

            embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = keras.Model(input_tensor, embedded_tensor)


if __name__ == "__main__":
    pass
