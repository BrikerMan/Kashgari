# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: demo.py
@time: 2019-01-19 16:37

"""
import kashgari
from kashgari.data.data_reader import load_data_from_csv_for_classifier
from kashgari.tasks.classification import CNN_LSTM_Model

if __name__ == "__main__":
    model = kashgari.tasks.classification.CNN_LSTM_Model()
    m = CNN_LSTM_Model()
    file_path = '/Users/brikerman/Desktop/ailab/Kashgari/kashgari/data/dataset.csv'
    x, y = load_data_from_csv_for_classifier(file_path)
    m.fit(x, y, batch_size=64)
    print("Hello world")
