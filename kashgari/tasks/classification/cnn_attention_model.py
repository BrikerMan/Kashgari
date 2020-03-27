# encoding: utf-8

# author: Adline
# contact: gglfxsld@gmail.com
# blog: https://medium.com/@Adline125

# file: cnn_attention_model.py
# time: 3:05 下午
from abc import ABC
from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.abc_model import ABCClassificationModel

L = keras.layers


class CNN_Attention_Model(ABCClassificationModel, ABC):

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv_layer1': {
                'filters': 264,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'conv_layer2': {
                'filters': 128,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'conv_layer3': {
                'filters': 64,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model
        # Query embeddings of shape [batch_size, Tq, dimension].
        query_embeddings = embed_model.output
        # Value embeddings of shape [batch_size, Tv, dimension].
        value_embeddings = embed_model.output

        # CNN layer.
        cnn_layer_1 = keras.layers.Conv1D(**config['conv_layer1'])
        # Query encoding of shape [batch_size, Tq, filters].
        query_seq_encoding = cnn_layer_1(query_embeddings)
        # Value encoding of shape [batch_size, Tv, filters].
        value_seq_encoding = cnn_layer_1(value_embeddings)

        cnn_layer_2 = keras.layers.Conv1D(**config['conv_layer2'])
        query_seq_encoding = cnn_layer_2(query_seq_encoding)
        value_seq_encoding = cnn_layer_2(value_seq_encoding)

        cnn_layer_3 = keras.layers.Conv1D(**config['conv_layer3'])
        query_seq_encoding = cnn_layer_3(query_seq_encoding)
        value_seq_encoding = cnn_layer_3(value_seq_encoding)

        # Query-value attention of shape [batch_size, Tq, filters].
        query_value_attention_seq = keras.layers.Attention()(
            [query_seq_encoding, value_seq_encoding])

        # Reduce over the sequence axis to produce encodings of shape
        # [batch_size, filters].
        query_encoding = keras.layers.Flatten()(
            query_seq_encoding)
        query_value_attention = keras.layers.Flatten()(
            query_value_attention_seq)

        # Concatenate query and document encodings to produce a DNN input layer.
        input_layer = keras.layers.Concatenate()(
            [query_encoding, query_value_attention])

        # Add DNN layers, and create Model.
        output = keras.layers.Dense(output_dim, **config['activation_layer'])(input_layer)

        self.tf_model = keras.Model(embed_model.input, output)
        self.tf_model.summary()


if __name__ == "__main__":
    pass
