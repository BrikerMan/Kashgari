# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_classification_model.py
# time: 2019-05-22 11:23

import random
import logging
from typing import Dict, Any, Tuple
from kashgari.tasks.base_model import BaseModel
from sklearn import metrics


class BaseClassificationModel(BaseModel):
    __task__ = "classification"

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def build_model_arc(self):
        raise NotImplementedError

    def evaluate(self,
                 x_data,
                 y_data,
                 batch_size=None,
                 digits=4,
                 debug_info=False) -> Tuple[float, float, Dict]:
        y_pred = self.predict(x_data, batch_size=batch_size)
        report = metrics.classification_report(y_data, y_pred, output_dict=True, digits=digits)
        print(metrics.classification_report(y_data, y_pred, digits=digits))
        if debug_info:
            for index in random.sample(list(range(len(x_data))), 5):
                logging.debug('------ sample {} ------'.format(index))
                logging.debug('x      : {}'.format(x_data[index]))
                logging.debug('y      : {}'.format(y_data[index]))
                logging.debug('y_pred : {}'.format(y_pred[index]))
        return report


if __name__ == "__main__":
    print("Hello world")
