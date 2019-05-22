# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: macros.py
@time: 2019-05-17 11:38

"""
import os
from pathlib import Path

DATA_PATH = os.path.join(str(Path.home()), '.kashgari')

Path(DATA_PATH).mkdir(exist_ok=True, parents=True)


class TaskType(object):
    CLASSIFICATION = 'classification'
    LABELING = 'labeling'


if __name__ == "__main__":
    print("Hello world")
