#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/8/29 11:47 上午
# File    : labeling.py
# Project : Kashgari

import os
from datetime import datetime
from typing import Type

import tensorflow as tf

from kashgari.callbacks import EvalCallBack
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import ABCLabelingModel
from kashgari.tasks.labeling import ALL_MODELS
from test_performance.classifications import ClassificationPerformance
from examples.tools import get_bert_path

log_root = "tf_dir/labeling/" + datetime.now().strftime("%m%d-%H:%M")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class LabelingPerformance(ClassificationPerformance):
    MODELS = ALL_MODELS

    def run_with_model_class(self, model_class: Type[ABCLabelingModel], epochs: int):
        bert_path = get_bert_path()

        train_x, train_y = ChineseDailyNerCorpus.load_data('train')
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        test_x, test_y = ChineseDailyNerCorpus.load_data('test')

        bert_embed = BertEmbedding(bert_path)
        model = model_class(bert_embed)

        log_path = os.path.join(log_root, model_class.__name__)
        file_writer = tf.summary.create_file_writer(log_path + "/metrics")
        file_writer.set_as_default()
        callbacks = [EvalCallBack(model, test_x, test_y, step=1, truncating=True)]
        # callbacks = []
        model.fit(train_x, train_y, valid_x, valid_y, epochs=epochs, callbacks=callbacks)

        report = model.evaluate(test_x, test_y)
        del model
        del bert_embed
        return report


if __name__ == '__main__':
    p = LabelingPerformance()
    p.run(epochs=10)
