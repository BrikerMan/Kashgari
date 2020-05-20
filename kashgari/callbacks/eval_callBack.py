# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: eval_callBack.py
# time: 6:53 下午

from typing import List, Any, Dict

import tensorflow as tf
from tensorflow import keras

from kashgari.tasks.abs_task_model import ABCTaskModel


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self,
                 kash_model: ABCTaskModel,
                 x_data: List[Any],
                 y_data: List[Any],
                 *,
                 step: int = 5,
                 truncating: bool = False,
                 batch_size: int = 256) -> None:
        """
        Evaluate callback, calculate precision, recall and f1
        Args:
            kash_model: the kashgari task model to evaluate
            x_data: feature data for evaluation
            y_data: label data for evaluation
            step: step, default 5
            truncating: truncating: remove values from sequences larger than `model.embedding.sequence_length`
            batch_size: batch size, default 256
        """
        super(EvalCallBack, self).__init__()
        self.kash_model: ABCTaskModel = kash_model
        self.x_data = x_data
        self.y_data = y_data
        self.step = step
        self.truncating = truncating
        self.batch_size = batch_size
        self.logs: List[Dict] = []

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        if (epoch + 1) % self.step == 0:
            report = self.kash_model.evaluate(self.x_data,  # type: ignore
                                              self.y_data,
                                              truncating=self.truncating,
                                              batch_size=self.batch_size)

            self.logs.append({
                'precision': report['precision'],
                'recall': report['recall'],
                'f1-score': report['f1-score']
            })

            tf.summary.scalar('eval f1-score', data=report['f1-score'], step=epoch)
            tf.summary.scalar('eval recall', data=report['recall'], step=epoch)
            tf.summary.scalar('eval precision', data=report['precision'], step=epoch)
            print(f"\nepoch: {epoch} precision: {report['precision']:.6f},"
                  f" recall: {report['recall']:.6f}, f1-score: {report['f1-score']:.6f}")
