# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 2019-05-22 12:40


from kashgari.tasks.classification.models import BiLSTM_Model
from kashgari.tasks.classification.models import CNN_Model
from kashgari.tasks.classification.models import CNN_LSTM_Model


BLSTMModel = BiLSTM_Model
CNNModel = CNN_Model
CNNLSTMModel = CNN_LSTM_Model