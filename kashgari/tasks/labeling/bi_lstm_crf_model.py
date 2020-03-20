# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bi_lstm_crf_model.py
# time: 4:09 下午

from typing import Dict, Any

from tensorflow import keras

import kashgari
from bert4keras.layers import ConditionalRandomField
from kashgari.layers import L
from kashgari.tasks.labeling.abc_model import ABCLabelingModel

kashgari.custom_objects['ConditionalRandomField'] = ConditionalRandomField


class BiLSTM_CRF_Model(ABCLabelingModel):
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
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        crf = ConditionalRandomField(output_dim, name='layer_crf')

        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_blstm']), name='layer_blstm'),
            L.Dropout(**config['layer_dropout'], name='layer_dropout'),
            L.Dense(**config['layer_dense'], name='layer_dense'),
            L.Dense(output_dim, name='layer_crf_dense'),
            crf
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.layer_crf = crf
        self.tf_model = keras.Model(embed_model.inputs, tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.dense_loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.dense_accuracy]
        super(BiLSTM_CRF_Model, self).compile_model(**kwargs)


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus

    x, y = ChineseDailyNerCorpus.load_data('valid')
    model = BiLSTM_CRF_Model()
    model.fit(x, y, epochs=10)
    print(model.info())
    model.save('./model')
