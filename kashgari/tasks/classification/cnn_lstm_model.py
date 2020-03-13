# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: cnn_lstm_model.py
# time: 4:06 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.abc_model import ABCClassificationModel

L = keras.layers


class BiLSTM_Model(ABCClassificationModel):
    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = self.label_processor.vocab_size

        config = self.get_default_hyper_parameters()
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level='DEBUG')

    from kashgari.embeddings import WordEmbedding
    w2v_path = '/Users/brikerman/Desktop/nlp/language_models/w2v/sgns.weibo.bigram-char'
    w2v = WordEmbedding(w2v_path, w2v_kwargs={'limit': 10000})

    from kashgari.corpus import SMP2018ECDTCorpus
    x, y = SMP2018ECDTCorpus.load_data()

    model = BiLSTM_Model(embedding=w2v)
    model.fit(x, y)

