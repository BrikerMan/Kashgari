# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bi_gru_model.py
# time: 4:37 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel


class BiGRU_Model(ABCClassificationModel):

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_gru': {
                'units': 128,
                'return_sequences': False
            },
            'layer_output': {
            }
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_stack = [
            L.Bidirectional(L.GRU(**config['layer_bi_gru'])),
            L.Dense(output_dim, **config['layer_output']),
            self._activation_layer()
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)


if __name__ == "__main__":
    from kashgari.corpus import SMP2018ECDTCorpus

    train_x, train_y = SMP2018ECDTCorpus.load_data()
    valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

    train_x = train_x * 10
    train_y = train_y * 10
    model = BiGRU_Model()
    from kashgari.generators import CorpusGenerator
    # train_gen = CorpusGenerator(train_x, train_y)
    model.fit(train_x, train_y, epochs=1)
    import time

    s = time.time()
    model.fit(train_x, train_y, epochs=5)
    print("Spend 1: ", time.time() - s)

    s = time.time()
    model.fit(train_x, train_y, epochs=5, use_tfdata=True)
    print("Spend 2: ", time.time() - s)

    y = model.predict(train_x[:20], debug_info=True)
    print(y)
    print(train_y[:20])

    test_x, test_y = SMP2018ECDTCorpus.load_data('test')
    model.evaluate(test_x, test_y)

    model.evaluate(valid_x, valid_y)
