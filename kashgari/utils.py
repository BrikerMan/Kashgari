# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: helpers.py
@time: 2019-05-17 11:37

"""
import os
import pathlib
import random
from typing import List, Tuple, Optional, Any


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def get_list_subset(target: List, index_list: List[int]) -> List:
    return [target[i] for i in index_list if i < len(target)]


def get_tuple_item(data: Optional[Tuple], index: int) -> Optional[Any]:
    if data and len(data) > index:
        return data[index]
    else:
        return None


def get_project_path() -> str:
    here = pathlib.Path(__file__).parent
    return os.path.abspath(os.path.join(here, '../'))


def wrap_as_tuple(original) -> Tuple[Any]:
    if isinstance(original, tuple):
        return original
    elif isinstance(original, list) or len(original) == 1:
        return tuple([original])
    return original


if __name__ == "__main__":
    print(get_project_path())
