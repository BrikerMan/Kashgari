# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: cnn_lstm_model.py
# time: 4:06 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel


class BiLSTM_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
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

        # build model structure in sequent way
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bi_lstm'])),
            L.Dense(output_dim, **config['layer_output']),
            self._activation_layer()
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model: keras.Model = keras.Model(embed_model.inputs, tensor)


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

    # 或者集成 CorpusGenerator 实现自己的数据迭代器
    # train_gen = CorpusGenerator()
    # model.fit_generator(train_gen=train_gen,
    #                     valid_gen=valid_gen,
    #                     batch_size=batch_size,
    #                     epochs=epochs)
