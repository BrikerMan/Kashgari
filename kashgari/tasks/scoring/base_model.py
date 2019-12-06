# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 11:36 上午


from typing import Dict, Any, Tuple

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
                 digits=4,
                 debug_info=False) -> Tuple[float, float, Dict]:
        """
        Build a text report showing the main classification metrics.

        Args:
            x_data:
            y_data:
            batch_size:
            digits:
            debug_info:

        Returns:

        """
        y_pred = self.predict(x_data, batch_size=batch_size)
        y_true = [seq[:len(y_pred[index])] for index, seq in enumerate(y_data)]
        mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
        r2_score = metrics.r2_score(y_true, y_pred)
        data = {
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score
        }
        return mean_squared_error, r2_score, data


if __name__ == "__main__":
    pass