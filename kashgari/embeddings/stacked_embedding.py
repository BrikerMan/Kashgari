# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: stacked_embedding.py
# time: 2019-05-23 09:18

from typing import Union, Optional, Tuple, List

import numpy as np
import tensorflow as tf

import kashgari
from kashgari.embeddings.base_embedding import Embedding
from kashgari.processors.base_processor import BaseProcessor
from kashgari.layers import L


# Todo: A better name for this class
class StackedEmbedding(Embedding):
    """Embedding layer without pre-training, train embedding layer while training model"""

    def __init__(self,
                 embeddings: List[Embedding],
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
        task = kashgari.CLASSIFICATION
        if all(isinstance(embed.sequence_length, tuple) for embed in embeddings):
            sequence_length = tuple([embed.sequence_length[0] for embed in embeddings])
        else:
            raise ValueError('Need to set sequence length for all embeddings while using stacked embedding')

        super(StackedEmbedding, self).__init__(task=task,
                                               sequence_length=sequence_length,
                                               embedding_size=embedding_size,
                                               processor=processor)
        self.embeddings = embeddings
        self.processor = embeddings[0].processor
        self._build_model()

    def _build_model(self, **kwargs):
        if self.embed_model is None and all(embed.embed_model is not None for embed in self.embeddings):
            layer_concatenate = L.Concatenate(name='layer_concatenate')
            inputs = [embed.embed_model.input for embed in self.embeddings]
            outputs = layer_concatenate([embed.embed_model.output for embed in self.embeddings])

            self.embed_model = tf.keras.Model(inputs, outputs)

    def analyze_corpus(self,
                       x: Union[Tuple[List[List[str]], ...], List[List[str]]],
                       y: Union[List[List[str]], List[str]]):
        for index in range(len(x)):
            self.embeddings[index].analyze_corpus(x[index], y)
        self._build_model()

    def process_x_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        """
        batch process feature data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        result = []
        for index, dataset in enumerate(data):
            result.append(self.embeddings[index].process_x_dataset((dataset,), subset))
        return tuple(result)


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus
    from kashgari.embeddings import BareEmbedding, NumericFeaturesEmbedding

    text, label = SMP2018ECDTCorpus.load_data()
    is_bold = np.random.randint(1, 3, (len(text), 12))

    text_embedding = BareEmbedding(task=kashgari.CLASSIFICATION,
                                   sequence_length=12)
    num_feature_embedding = NumericFeaturesEmbedding(2,
                                                     'is_bold',
                                                     sequence_length=12)

    stack_embedding = StackedEmbedding([text_embedding, num_feature_embedding])
    stack_embedding.analyze_corpus((text, is_bold), label)

    r = stack_embedding.embed((text[:3], is_bold[:3]))
    print(r)
    print(r.shape)

    import kashgari
    from kashgari.embeddings import NumericFeaturesEmbedding, BareEmbedding, StackedEmbedding

    text = ['NLP', 'Projects', 'Project', 'Name', ':']
    start_of_p = [1, 2, 1, 2, 2]
    bold = [1, 1, 1, 1, 2]
    center = [1, 1, 2, 2, 2]
    label = ['B-Category', 'I-Category', 'B-ProjectName', 'I-ProjectName', 'I-ProjectName']

    text_list = [text] * 100
    start_of_p_list = [start_of_p] * 100
    bold_list = [bold] * 100
    center_list = [center] * 100
    label_list = [label] * 100

    # You can use WordEmbedding or BERTEmbedding for your text embedding
    SEQUENCE_LEN = 100
    text_embedding = BareEmbedding(task=kashgari.LABELING, sequence_length=SEQUENCE_LEN)
    start_of_p_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                    feature_name='start_of_p',
                                                    sequence_length=SEQUENCE_LEN)

    bold_embedding = NumericFeaturesEmbedding(feature_count=2,
                                              feature_name='bold',
                                              sequence_length=SEQUENCE_LEN)

    center_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                feature_name='center',
                                                sequence_length=SEQUENCE_LEN)

    # first one must be the text, embedding
    stack_embedding = StackedEmbedding([
        text_embedding,
        start_of_p_embedding,
        bold_embedding,
        center_embedding
    ])

    x = (text_list, start_of_p_list, bold_list, center_list)
    y = label_list
    stack_embedding.analyze_corpus(x, y)

    # Now we can embed with this stacked embedding layer
    print(stack_embedding.embed(x))

    # We can build any labeling model with this embedding

    from kashgari.tasks.labeling import BLSTMModel

    model = BLSTMModel(embedding=stack_embedding)
    model.fit(x, y, epochs=1)

    print(model.predict(x))
    print(model.predict_entities(x))

    from tensorflow.python import keras

    keras.utils.plot_model(model.tf_model, to_file='model.png', show_shapes=True)
