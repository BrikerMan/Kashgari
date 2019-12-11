# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 11:36 上午


from typing import Callable
from typing import Dict, Any

import numpy as np
from sklearn import metrics

from kashgari.tasks.base_model import BaseModel


class BaseScoringModel(BaseModel):
    """Base Sequence Labeling Model"""

    __task__ = 'scoring'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = 'mse'
        if kwargs.get('optimizer') is None:
            kwargs['optimizer'] = 'rmsprop'
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = ['mae']
        super(BaseScoringModel, self).compile_model(**kwargs)

    def evaluate(self,
                 x_data,
                 y_data,
                 batch_size=None,
                 should_round: bool = False,
                 round_func: Callable = None,
                 digits=4,
                 debug_info=False) -> Dict:
        """
        Build a text report showing the main classification metrics.

        Args:
            x_data:
            y_data:
            batch_size:
            should_round:
            round_func:
            digits:
            debug_info:

        Returns:

        """
        y_pred = self.predict(x_data, batch_size=batch_size)

        if should_round:
            if round_func is None:
                round_func = np.round
            print(self.processor.output_dim)
            if self.processor.output_dim != 1:
                raise ValueError('Evaluate with round function only accept 1D output')
            y_pred = [round_func(i) for i in y_pred]
            report = metrics.classification_report(y_data,
                                                   y_pred,
                                                   digits=digits)

            report_dic = metrics.classification_report(y_data,
                                                       y_pred,
                                                       output_dict=True,
                                                       digits=digits)
            print(report)
        else:
            mean_squared_error = metrics.mean_squared_error(y_data, y_pred)
            r2_score = metrics.r2_score(y_data, y_pred)
            report_dic = {
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score
            }
            print(f"mean_squared_error : {mean_squared_error}\n"
                  f"r2_score           : {r2_score}")
        return report_dic


if __name__ == "__main__":
    pass
