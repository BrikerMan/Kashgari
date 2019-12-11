# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: blstm_model.py
# time: 12:17 下午
import os
import tempfile
import time
import unittest
import kashgari
import numpy as np

from tests.corpus import NERCorpus
from kashgari.tasks.scoring import BiLSTM_Model


class TestBiLSTM_Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = BiLSTM_Model

    def test_basic_use_build(self):
        x, _ = NERCorpus.load_corpus()
        y = np.random.random((len(x),))
        model = self.model_class()
        model.fit(x, y, epochs=1)
        res = model.predict(x[:20])
        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)

        pd_model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        kashgari.utils.convert_to_saved_model(model,
                                              pd_model_path)

        new_model = kashgari.utils.load_model(model_path)
        new_res = new_model.predict(x[:20])
        assert np.array_equal(new_res, res)

        new_model.compile_model()
        model.fit(x, y, x, y, epochs=1)
        model.evaluate(x, y)

        rounded_y = np.round(y)
        model.evaluate(x, rounded_y, should_round=True)

    def test_multi_output(self):
        x, _ = NERCorpus.load_corpus()
        y = np.random.random((len(x), 4))
        model = self.model_class()
        model.fit(x, y, x, y, epochs=1)
        with self.assertRaises(ValueError):
            model.evaluate(x, y, should_round=True)


if __name__ == "__main__":
    pass
