#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/2 12:12 下午
# File    : dependency_check.py
# Project : Kashgari

import tensorflow as tf

from distutils.version import LooseVersion


def dependency_check() -> None:
    if LooseVersion(tf.__version__) < '2.2.0':
        try:
            import tensorflow_addons as tfa
            if LooseVersion(tfa.__version__) > '0.10.0':
                raise ImportError("TF 2.1 required lower version of tensorflow_addons, "
                                  "install using `$pip install tensorflow_addons<=0.10.0`")
        except ImportError:
            raise ImportError("TF 2.1 required lower version of tensorflow_addons, "
                              "install using `$pip install tensorflow_addons<=0.10.0`")
