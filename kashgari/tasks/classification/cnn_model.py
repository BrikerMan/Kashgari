# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: cnn_model.py
# time: 3:31 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.abc_model import ABCClassificationModel

L = keras.layers


class CNN_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv1d_layer': {
                'filters': 128,
                'kernel_size': 5,
                'activation': 'relu'
            },
            'max_pool_layer': {},
            'dense_layer': {
                'units': 64,
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

        # build model structure in sequent way
        layers_seq = [
            L.Conv1D(**config['conv1d_layer']),
            L.GlobalMaxPooling1D(**config['max_pool_layer']),
            L.Dense(**config['dense_layer']),
            L.Dense(output_dim, **config['activation_layer'])
        ]

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)


if __name__ == "__main__":
    pass
