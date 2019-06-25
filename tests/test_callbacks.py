# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_utils.py
# time: 14:46

import unittest
from kashgari import callbacks
from kashgari.tasks.classification import BLSTMModel
from kashgari.tasks.labeling import BiLSTM_Model as Labeling_BiLSTM_Model
from kashgari.corpus import ChineseDailyNerCorpus, SMP2018ECDTCorpus


class TestCallbacks(unittest.TestCase):

    def test_labeling_eval_callback(self):
        train_x, train_y = ChineseDailyNerCorpus.load_data()
        test_x, test_y = ChineseDailyNerCorpus.load_data('test')

        train_x = train_x[:1000]
        train_y = train_y[:1000]
        model = Labeling_BiLSTM_Model()
        eval_callback = callbacks.EvalCallBack(model, test_x, test_y, step=1)
        model.fit(train_x, train_y, callbacks=[eval_callback], epochs=1)

    def test_classification_eval_callback(self):
        train_x, train_y = SMP2018ECDTCorpus.load_data()
        test_x, test_y = SMP2018ECDTCorpus.load_data('test')

        train_x = train_x[:1000]
        train_y = train_y[:1000]
        model = BLSTMModel()
        eval_callback = callbacks.EvalCallBack(model, test_x, test_y, step=1)
        model.fit(train_x, train_y, callbacks=[eval_callback], epochs=1)


if __name__ == "__main__":
    print("hello, world")