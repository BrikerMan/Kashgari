# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 2019-05-20 11:34

from kashgari.tasks.labeling.models import CNN_LSTM_Model

from kashgari.tasks.labeling.models import BiLSTM_Model
from kashgari.tasks.labeling.models import BiLSTM_CRF_Model

from kashgari.tasks.labeling.models import BiGRU_Model
from kashgari.tasks.labeling.models import BiGRU_CRF_Model

if __name__ == "__main__":
    print("Hello world")
