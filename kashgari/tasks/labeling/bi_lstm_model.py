# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bi_lstm_model.py
# time: 4:36 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.layers import L
from kashgari.tasks.labeling.abc_model import ABCLabelingModel


class BiLSTM_Model(ABCLabelingModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_blstm']), name='layer_blstm'),
            L.Dropout(**config['layer_dropout'], name='layer_dropout'),
            L.Dense(output_dim, **config['layer_time_distributed']),
            L.Activation(**config['layer_activation'])
        ]
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus

    x, y = ChineseDailyNerCorpus.load_data()
    x_valid, y_valid = ChineseDailyNerCorpus.load_data('valid')
    model = BiLSTM_Model()
    model.fit(x, y, x_valid, y_valid, epochs=2)
    model.evaluate(*ChineseDailyNerCorpus.load_data('test'))
