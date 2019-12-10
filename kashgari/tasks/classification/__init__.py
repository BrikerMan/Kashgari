# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: __init__.py
# time: 2019-05-22 12:40


from kashgari.tasks.classification.models import BiLSTM_Model
from kashgari.tasks.classification.models import BiGRU_Model
from kashgari.tasks.classification.models import CNN_Model
from kashgari.tasks.classification.models import CNN_LSTM_Model
from kashgari.tasks.classification.models import CNN_GRU_Model
from kashgari.tasks.classification.models import AVCNN_Model
from kashgari.tasks.classification.models import KMax_CNN_Model
from kashgari.tasks.classification.models import R_CNN_Model
from kashgari.tasks.classification.models import AVRNN_Model
from kashgari.tasks.classification.models import Dropout_BiGRU_Model
from kashgari.tasks.classification.models import Dropout_AVRNN_Model
from kashgari.tasks.classification.dpcnn_model import DPCNN_Model


BLSTMModel = BiLSTM_Model
BGRUModel = BiGRU_Model
CNNModel = CNN_Model
CNNLSTMModel = CNN_LSTM_Model
CNNGRUModel = CNN_GRU_Model
AVCNNModel = AVCNN_Model
KMaxCNNModel = KMax_CNN_Model
RCNNModel = R_CNN_Model
AVRNNModel = AVRNN_Model
DropoutBGRUModel = Dropout_BiGRU_Model
DropoutAVRNNModel = Dropout_AVRNN_Model

DPCNN = DPCNN_Model
