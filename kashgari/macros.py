# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: macros.py
# time: 12:37 下午

import os
import logging
from pathlib import Path
import tensorflow as tf

DATA_PATH = os.path.join(str(Path.home()), '.kashgari')

Path(DATA_PATH).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    pass
