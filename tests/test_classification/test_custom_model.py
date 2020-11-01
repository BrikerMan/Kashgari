# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_custom_model.py
# time: 6:07 下午

import unittest
from typing import Dict, Any

from tensorflow import keras

import tests.test_classification.test_bi_lstm_model as base
from kashgari.embeddings import WordEmbedding
from kashgari.layers import L
from kashgari.tasks.classification.abc_model import ABCClassificationModel
from tests.test_macros import TestMacros


class Double_BiLSTM_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Any]:
        return {
            'layer_lstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_lstm2': {
                'units': 64,
                'return_sequences': False
            },
            'layer_dropout': {
                'rate': 0.5
            },
            'layer_output': {

            }
        }

    def build_model_arc(self) -> None:
        config = self.hyper_parameters
        output_dim = self.label_processor.vocab_size
        embed_model = self.embedding.embed_model

        # 定义模型架构
        self.tf_model = keras.Sequential([
            embed_model,
            L.Bidirectional(L.LSTM(**config['layer_lstm1'])),
            L.Bidirectional(L.LSTM(**config['layer_lstm2'])),
            L.Dropout(**config['layer_dropout']),
            L.Dense(output_dim, **config['layer_output']),
            self._activation_layer()
        ])


class TestCustom_Model(base.TestBiLSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = Double_BiLSTM_Model
        cls.w2v_embedding = WordEmbedding(TestMacros.w2v_path)


if __name__ == "__main__":
    unittest.main()
