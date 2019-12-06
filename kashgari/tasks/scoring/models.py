# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 11:38 上午


import logging
from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.scoring.base_model import BaseScoringModel
from kashgari.layers import L


class BiLSTM_Model(BaseScoringModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'linear'
            }
        }

    def build_model_arc(self):
        output_dim = self.processor.output_dim
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus
    import numpy as np

    x, y = SMP2018ECDTCorpus.load_data('valid')
    y = np.random.random((len(x), 4))
    model = BiLSTM_Model()
    model.fit(x, y)
    print(model.predict(x[:10]))

