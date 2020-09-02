#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/8/29 11:47 上午
# File    : labeling.py
# Project : Kashgari

import os
from typing import Type

import wandb

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import ABCLabelingModel
from kashgari.tasks.labeling import ALL_MODELS
from test_performance.classifications import ClassificationPerformance, WandbCallback
from test_performance.tools import get_bert_path

os.environ["WANDB_RUN_GROUP"] = "labeling_run_" + wandb.util.generate_id()


class LabelingPerformance(ClassificationPerformance):
    MODELS = ALL_MODELS

    def run_with_model_class(self, model_class: Type[ABCLabelingModel], epochs: int):
        bert_path = get_bert_path()

        train_x, train_y = ChineseDailyNerCorpus.load_data('train')
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        test_x, test_y = ChineseDailyNerCorpus.load_data('test')
        wandb.init(project="kashgari",
                   name=model_class.__name__,
                   reinit=True,
                   tags=["bert", "classification"])

        bert_embed = BertEmbedding(bert_path)
        model = model_class(bert_embed)
        callbacks = [WandbCallback(model, test_x, test_y)]
        # callbacks = []
        model.fit(train_x, train_y, valid_x, valid_y, epochs=epochs, callbacks=callbacks)

        report = model.evaluate(test_x, test_y)
        del model
        del bert_embed
        return report


if __name__ == '__main__':
    p = LabelingPerformance()
    p.run(epochs=10)
