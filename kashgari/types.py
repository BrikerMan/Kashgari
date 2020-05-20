# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: types.py
# time: 3:54 下午

from typing import List, Union, Tuple

TextSamplesVar = List[List[str]]
NumSamplesListVar = List[List[int]]
LabelSamplesVar = Union[TextSamplesVar, List[str]]

ClassificationLabelVar = List[str]
MultiLabelClassificationLabelVar = Union[List[List[str]], List[Tuple[str]]]

if __name__ == "__main__":
    pass
