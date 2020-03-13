# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: utils.py
# time: 12:39 下午

import random
from types import TracebackType
from typing import List, Generator, Type, Optional


def get_list_subset(target: List, index_list: List[int]) -> List:
    return [target[i] for i in index_list if i < len(target)]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


if __name__ == "__main__":
    pass
