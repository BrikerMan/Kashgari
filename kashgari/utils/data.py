# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: data.py
# time: 11:22 上午

import random
from typing import List, Union, TypeVar, Tuple

import numpy as np

T = TypeVar("T")


def get_list_subset(target: List[T], index_list: List[int]) -> List[T]:
    """
    Get the subset of the target list
    Args:
        target: target list
        index_list: subset items index

    Returns:
        subset of the original list
    """
    return [target[i] for i in index_list if i < len(target)]


def unison_shuffled_copies(a: List[T],
                           b: List[T]) -> Union[Tuple[List[T], ...], Tuple[np.ndarray, ...]]:
    """
    Union shuffle two arrays
    Args:
        a:
        b:

    Returns:

    """
    data_type = type(a)
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    if data_type == np.ndarray:
        return np.array(a), np.array(b)
    return list(a), list(b)
