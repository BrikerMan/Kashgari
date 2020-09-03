# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_gru_model.py
# time: 12:35 上午

import unittest

import tests.test_labeling.test_bi_lstm_model as base
from kashgari.tasks.labeling import BiGRU_Model


class TestBiGRU_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiGRU_Model

    def test_basic_use(self):
        super(TestBiGRU_Model, self).test_basic_use()

    def test_predict_and_callback(self):
        from kashgari.corpus import ChineseDailyNerCorpus
        from kashgari.callbacks import EvalCallBack

        train_x, train_y = ChineseDailyNerCorpus.load_data('train')
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        model = BiGRU_Model(sequence_length=10)

        eval_callback = EvalCallBack(kash_model=model,
                                     x_data=valid_x[:200],
                                     y_data=valid_y[:200],
                                     truncating=True,
                                     step=1)

        model.fit(train_x[:300], train_y[:300],
                  valid_x[:200], valid_y[:200],
                  epochs=1,
                  callbacks=[eval_callback])
        response = model.predict(train_x[:200], truncating=True)
        lengths = [len(i) for i in response]
        assert all([(i <= 10) for i in lengths])

        response = model.predict(train_x[:200])
        lengths = [len(i) for i in response]
        assert not all([(i <= 10) for i in lengths])


if __name__ == "__main__":
    pass
