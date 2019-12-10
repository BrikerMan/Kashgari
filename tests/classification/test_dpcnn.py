# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_dpcnn.py
# time: 2019-07-02 20:45

import tests.classification.test_bi_lstm as base
from kashgari.tasks.classification import DPCNN_Model


class TestDPCNN_Model(base.TestBi_LSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.epochs = 1
        cls.model_class = DPCNN_Model


    def test_custom_hyper_params(self):
        hyper_params = self.model_class.get_default_hyper_parameters()

        for layer, config in hyper_params.items():
            for key, value in config.items():
                if isinstance(value, bool):
                    pass
                elif isinstance(value, int):
                    hyper_params[layer][key] = value + 15 if value >= 64 else value
        model = self.model_class(embedding=base.w2v_embedding,
                                 hyper_parameters=hyper_params)
        model.fit(base.valid_x, base.valid_y, epochs=1)
        assert True

if __name__ == "__main__":
    print("Hello world")
