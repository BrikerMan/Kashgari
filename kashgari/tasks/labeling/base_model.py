# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_model.py
# time: 2019-05-20 13:07


from typing import Dict, Any, Tuple

import random
import logging
import numpy as np
from kashgari.loss import weighted_categorical_crossentropy
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
                         debug_info=False):
        """Gets entities from sequence.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            join_chunk: str or False,
            debug_info: Bool, Should print out the logging info.

        Returns:
            list: list of entity.

        Example:
            >>> from seqeval.metrics.sequence_labeling import get_entities
            >>> seq = 'President Obama is speaking at the White House.'
            >>> model.predict_entities([seq])
            [[
            {'entity': 'PER', 'start': 0, 'end': 1, 'value': ['President', 'Obama']},
            {'entity': 'LOC', 'start': 6, 'end': 7, 'value': ['White', 'House']}
            ]]
        """
        if isinstance(x_data, tuple):
            text_seq = x_data[0]
        else:
            text_seq = x_data
        res = self.predict(x_data, batch_size, debug_info)
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
            final_res.append(seq_data)
        return final_res

    # Todo: Better way to do this, too
    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            idx2label = self.embedding.processor.idx2label
            weight = np.full((len(idx2label),), 50)
            for idx, label in idx2label.items():
                if label == self.embedding.processor.token_pad:
                    weight[idx] = 1
                if label in ['O']:
                    weight[idx] = 10
            weight_dict = {}
            for idx, label in idx2label.items():
                weight_dict[label] = weight[idx]
            logging.debug(f"label weights set to {weight_dict}")
            kwargs['loss'] = weighted_categorical_crossentropy(weight)
        super(BaseLabelingModel, self).compile_model(**kwargs)

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
        y_true = [seq[:self.embedding.sequence_length] for seq in y_data]

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
    from kashgari.tasks.labeling import BLSTMCRFModel
    from kashgari.corpus import ChineseDailyNerCorpus

    train_x, train_y = ChineseDailyNerCorpus.load_data('train')
    valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

    train_x, train_y = train_x[:5120], train_y[:5120]

    model = BLSTMCRFModel()
    model.build_model(train_x[:100], train_y[:100])

    model.fit(train_x, train_y, valid_x, valid_y, epochs=20)
    r = model.predict_entities(train_x[:5], join_chunk='')

    import pprint

    pprint.pprint(r)
    # model.evaluate(train_x[:20], train_y[:20])
    print("Hello world")
