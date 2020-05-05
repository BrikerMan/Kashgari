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

    def build_model_arc(self):
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
    from kashgari.corpus import JigsawToxicCommentCorpus
    corpus = JigsawToxicCommentCorpus('/Users/brikerman/Downloads/'
                                      'jigsaw-toxic-comment-classification-challenge/train.csv')
    x, y = corpus.load_data()
    model = BiGRU_Model(multi_label=True)
    from kashgari.generators import CorpusGenerator
    train_gen = CorpusGenerator(x, y)
    model.build_model(train_gen)
    model.tf_model.summary()
    model.fit(x, y, epochs=1)

    y = model.predict(x[:5], debug_info=True)
    print(y)
