# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: run_flask_api
@time: 2019-02-24

"""
import random
from flask import Flask, jsonify
from kashgari.tasks.classification import KMaxCNNModel
from kashgari.corpus import SMP2017ECDTClassificationCorpus

train_x, train_y = SMP2017ECDTClassificationCorpus.get_classification_data()

model = KMaxCNNModel()
model.fit(train_x, train_y)


app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def get_tasks():
    x = random.choice(train_x)
    y = model.predict(x, output_dict=True)
    return jsonify({'x': x, 'y': y})


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    model.predict(train_x[10])
    app.run(debug=True, port=8080)
