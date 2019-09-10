# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 2019-05-20 13:07


from typing import Dict, Any, Tuple

import random
import logging
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities

from kashgari.tasks.base_model import BaseModel


class BaseLabelingModel(BaseModel):
    """Base Sequence Labeling Model"""

    __task__ = 'labeling'

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def predict_entities(self,
                         x_data,
                         batch_size=None,
                         join_chunk=' ',
                         debug_info=False,
                         predict_kwargs: Dict = None):
        """Gets entities from sequence.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            join_chunk: str or False,
            debug_info: Bool, Should print out the logging info.
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            list: list of entity.
        """
        if isinstance(x_data, tuple):
            text_seq = x_data[0]
        else:
            text_seq = x_data
        res = self.predict(x_data, batch_size, debug_info, predict_kwargs)
        new_res = [get_entities(seq) for seq in res]
        final_res = []
        for index, seq in enumerate(new_res):
            seq_data = []
            for entity in seq:
                if join_chunk is False:
                    value = text_seq[index][entity[1]:entity[2] + 1],
                else:
                    value = join_chunk.join(text_seq[index][entity[1]:entity[2] + 1])

                seq_data.append({
                    "entity": entity[0],
                    "start": entity[1],
                    "end": entity[2],
                    "value": value,
                })
            final_res.append({
                'text': join_chunk.join(text_seq[index]),
                'text_raw': text_seq[index],
                'labels': seq_data
            })
        return final_res

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

        new_y_pred = []
        for x in y_pred:
            new_y_pred.append([str(i) for i in x])
        new_y_true = []
        for x in y_true:
            new_y_true.append([str(i) for i in x])

        if debug_info:
            for index in random.sample(list(range(len(x_data))), 5):
                logging.debug('------ sample {} ------'.format(index))
                logging.debug('x      : {}'.format(x_data[index]))
                logging.debug('y_true : {}'.format(y_true[index]))
                logging.debug('y_pred : {}'.format(y_pred[index]))
        report = classification_report(y_true, y_pred, digits=digits)
        print(classification_report(y_true, y_pred, digits=digits))
        return report

    def build_model_arc(self):
        raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from kashgari.tasks.labeling import BiLSTM_Model
    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.utils import load_model

    train_x, train_y = ChineseDailyNerCorpus.load_data('train', shuffle=False)
    valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

    train_x, train_y = train_x[:5120], train_y[:5120]

    model = load_model('/Users/brikerman/Desktop/blstm_model')
    # model.build_model(train_x[:100], train_y[:100])

    # model.fit(train_x[:1000], train_y[:1000], epochs=10)

    # model.evaluate(train_x[:20], train_y[:20])
    print("Hello world")
