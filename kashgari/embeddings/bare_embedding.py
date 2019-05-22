# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bare_embedding.py
# time: 2019-05-20 10:36
import logging
from typing import Union, Optional, Tuple

from tensorflow import keras

import kashgari.macros as k
from kashgari.embeddings.base_embedding import Embedding
from kashgari.pre_processors.base_processor import BaseProcessor

L = keras.layers


# Todo: A better name for this class
class BareEmbedding(Embedding):

    """Embedding layer without pre-training, train embedding layer while training model"""

    def __init__(self,
                 task: str = None,
                 sequence_length: Union[Tuple[int, ...], str, int] = 'auto',
                 embedding_size: int = 100,
                 processor: Optional[BaseProcessor] = None):
        """
        Init bare embedding (embedding without pre-training)

        Args:
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            embedding_size: Dimension of the dense embedding.
        """
        super(BareEmbedding, self).__init__(task=task,
                                            sequence_length=sequence_length,
                                            embedding_size=embedding_size,
                                            processor=processor)
        if processor:
            self._build_model()

    def _build_model(self, **kwargs):
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            input_layers = []
            output_layers = []
            for index, seq_len in enumerate(self.sequence_length):
                input_tensor = L.Input(shape=(seq_len,),
                                       name=f'input_{index}')
                layer_embedding = L.Embedding(self.token_count,
                                              self.embedding_size,
                                              name=f'layer_embedding_{index}')

                embedded_tensor = layer_embedding(input_tensor)

                input_layers.append(input_tensor)
                output_layers.append(embedded_tensor)
            if len(output_layers) > 1:
                layer_concatenate = L.Concatenate(name='layer_concatenate')
                output = layer_concatenate(output_layers)
            else:
                output = output_layers

            self.embed_model = keras.Model(input_layers, output)


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus
    from kashgari.pre_processors import ClassificationProcessor
    import kashgari

    x, y = SMP2018ECDTCorpus.load_data()
    p = ClassificationProcessor()
    p.analyze_corpus(x, y)

    embedding = BareEmbedding(task=kashgari.CLASSIFICATION,
                              sequence_length=12, processor=p)
    embedding._build_model()
    embedding.embed_model.summary()
    r = embedding.embed(x[:2])
    print(r)
    print(r.shape)
