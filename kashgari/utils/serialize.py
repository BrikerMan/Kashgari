# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: serialize.py
# time: 11:23 上午

import pydoc
from typing import Dict, Any


def load_data_object(data: Dict,
                     custom_objects: Dict = None,
                     **kwargs: Dict) -> Any:
    """
    Load Object From Dict
    Args:
        data:
        custom_objects:
        **kwargs:

    Returns:

    """
    if custom_objects is None:
        custom_objects = {}

    if data['__class_name__'] in custom_objects:
        obj: Any = custom_objects[data['__class_name__']](**data['config'], **kwargs)
    else:
        module_name = f"{data['__module__']}.{data['__class_name__']}"
        obj: Any = pydoc.locate(module_name)(**data['config'], **kwargs)  # type: ignore
    if hasattr(obj, '_override_load_model'):
        obj._override_load_model(data)

    return obj


if __name__ == "__main__":
    pass
