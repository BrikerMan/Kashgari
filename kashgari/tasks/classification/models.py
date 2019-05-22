# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-22 11:26

import logging
import tensorflow as tf
from typing import Dict, Any
from kashgari.layers import L
from kashgari.tasks.classification.base_model import BaseClassificationModel


class BLSTMModel(BaseClassificationModel):
    __architect_name__ = 'BLSTMModel'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 256,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    print(BLSTMModel.get_default_hyper_parameters())
    logging.basicConfig(level=logging.DEBUG)
    from kashgari.corpus import SMP2018ECDTCorpus

    x, y = SMP2018ECDTCorpus.load_data()

    m = BLSTMModel()
    m.build_model(x, y)
    r = m.get_data_generator(x, y)
    m.fit(x, y, epochs=5)
    m.evaluate(x, y)
    print(m.predict(x[:10]))
