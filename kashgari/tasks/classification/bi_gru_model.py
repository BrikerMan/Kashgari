# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bi_gru_model.py
# time: 4:37 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.abc_model import ABCClassificationModel

L = keras.layers


class BiGRU_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_gru': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = self.label_processor.vocab_size
        config = self.default_hyper_parameters()
        embed_model = self.embedding.embed_model

        layer_bi_gru = L.Bidirectional(L.GRU(**config['layer_bi_gru']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_gru(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    pass
