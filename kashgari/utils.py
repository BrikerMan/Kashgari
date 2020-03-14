# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: utils.py
# time: 12:39 下午

import random
import numpy as np
from typing import List
from typing import TypeVar, Tuple

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


def get_list_subset(target: List[T], index_list: List[int]) -> List[T]:
    return [target[i] for i in index_list if i < len(target)]


def unison_shuffled_copies(a: T1, b: T2) -> Tuple[T1, T2]:
    data_type = type(a)
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    if data_type == np.ndarray:
        return np.array(a), np.array(b)
    return list(a), list(b)


if __name__ == "__main__":
    pass
