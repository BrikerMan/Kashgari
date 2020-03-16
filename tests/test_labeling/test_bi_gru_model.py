# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bi_gru_model.py
# time: 12:35 上午

import unittest

from kashgari.tasks.classification import BiGRU_Model


class TestBiLSTM_Model(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = BiGRU_Model


if __name__ == "__main__":
    pass
