#!/usr/bin/env python
# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: setup.py
@time: 2019-01-24 16:42

"""
import pathlib

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'kashgari'
DESCRIPTION = 'simple and powerful state-of-the-art NLP framework with pre-trained word2vec and bert embedding.'
URL = 'https://github.com/BrikerMan/Kashgari'
EMAIL = 'eliyar917@gmail.com'
AUTHOR = 'BrikerMan'
LICENSE = 'Apache License 2.0'

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

required = [
    'Keras>=2.2.0',
    'keras_bert',
    'h5py>=2.7.1',
    'keras-bert==0.25.0',
    'scikit-learn>=0.19.1',
    'numpy>=1.14.3',
    'download>=0.3.3',
    'seqeval >=0.0.3',
    'colorlog>=4.0.0',
    'gensim>=3.5.0',
    # 'bz2file>=0.98',
    'sklearn',
    'pandas>=0.23.0'
]

# long_description = ""

setup(
    name=NAME,
    version='0.1.6',
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=required,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        # 'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)


if __name__ == "__main__":
    print("Hello world")
