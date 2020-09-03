#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/1 11:51 下午
# File    : bi_lstm_crf_model.py
# Project : Kashgari

from typing import Dict, Any

from tensorflow import keras
from kashgari.layers import L, ConditionalRandomField
from kashgari.tasks.labeling.abc_model import ABCLabelingModel


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
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        crf = ConditionalRandomField()

        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_blstm']), name='layer_blstm'),
            L.Dropout(**config['layer_dropout'], name='layer_dropout'),
            L.Dense(output_dim, **config['layer_time_distributed']),
            crf
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)
        self.crf_layer = crf

    def compile_model(self,
                      loss: Any = None,
                      optimizer: Any = None,
                      metrics: Any = None,
                      **kwargs: Any) -> None:
        if loss is None:
            loss = self.crf_layer.loss
        if metrics is None:
            metrics = [self.crf_layer.accuracy]
        super(BiLSTM_CRF_Model, self).compile_model(loss=loss,
                                                    optimizer=optimizer,
                                                    metrics=metrics,
                                                    **kwargs)
