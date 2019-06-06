# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py.py
@time: 2019-01-19 13:42

"""
import kashgari.embeddings
import kashgari.corpus
import kashgari.tasks

from kashgari.tasks import classification
from kashgari.tasks import seq_labeling

from kashgari.macros import config

from kashgari.version import __version__

if __name__ == "__main__":
    print("Hello world")
