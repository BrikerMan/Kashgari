#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/8/29 11:16 上午
# File    : classifications.py
# Project : Kashgari

import logging
import os
import time
from typing import Type

import pandas as pd
import wandb
from tensorflow.keras.callbacks import Callback

from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.classification import ABCClassificationModel
from kashgari.tasks.classification import ALL_MODELS
from test_performance.tools import get_bert_path

os.environ["WANDB_RUN_GROUP"] = "classification_run_" + wandb.util.generate_id()


class WandbCallback(Callback):
    def __init__(self, kash_model, test_x, test_y):
        self.kash_model: ABCClassificationModel = kash_model
        self.test_x = test_x
        self.test_y = test_y

    def on_epoch_end(self, epoch, logs=None):
        report = self.kash_model.evaluate(self.test_x, self.test_y)
        wandb.log({'epoch': epoch, 'precision': report['precision']}, step=epoch)
        wandb.log({'epoch': epoch, 'recall': report['recall']}, step=epoch)
        wandb.log({'epoch': epoch, 'f1-score': report['f1-score']}, step=epoch)


class ClassificationPerformance:
    MODELS = ALL_MODELS

    def run_with_model_class(self, model_class: Type[ABCClassificationModel], epochs: int):
        bert_path = get_bert_path()

        train_x, train_y = SMP2018ECDTCorpus.load_data('train')
        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        test_x, test_y = SMP2018ECDTCorpus.load_data('test')

        wandb.init(project="kashgari",
                   name=model_class.__name__,
                   reinit=True,
                   tags=["bert", "classification"])

        bert_embed = BertEmbedding(bert_path)
        model = model_class(bert_embed)

        callbacks = [WandbCallback(model, test_x, test_y)]

        model.fit(train_x, train_y, valid_x, valid_y, epochs=epochs, callbacks=callbacks)

        report = model.evaluate(test_x, test_y)
        del model
        del bert_embed
        return report

    def run(self, epochs=10):
        logging.basicConfig(level='DEBUG')
        reports = []
        for model_class in self.MODELS:
            logging.info("=" * 80)
            logging.info("")
            logging.info("")
            logging.info(f" Start Training {model_class.__name__}")
            logging.info("")
            logging.info("")
            logging.info("=" * 80)
            start = time.time()
            report = self.run_with_model_class(model_class, epochs=epochs)
            time_cost = time.time() - start
            reports.append({
                'model_name': model_class.__name__,
                "epoch": epochs,
                'f1-score': report['f1-score'],
                'precision': report['precision'],
                'recall': report['recall'],
                'time': f"{int(time_cost // 60):02}:{int(time_cost % 60):02}"
            })

        df = pd.DataFrame(reports)
        print(df.to_markdown())


if __name__ == '__main__':
    p = ClassificationPerformance()
    p.run()
