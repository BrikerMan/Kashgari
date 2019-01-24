# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: type_hints.py
@time: 2019-01-21 13:55

"""
from typing import Union, List

# ClassificationXType = Union[List[List[str]], List[str]]
# ClassificationYType = List[str]

TextSeqType = List[str]
TokenSeqType = List[int]

TextSeqInputType = Union[List[TextSeqType], TextSeqType]
TokenSeqInputType = Union[List[TokenSeqType], TokenSeqType]


if __name__ == "__main__":
    print("Hello world")
