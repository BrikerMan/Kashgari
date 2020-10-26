# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: model.py
# time: 10:57 上午

import json
import os
import pathlib
import time
from typing import Union, Any

from kashgari.tasks.abs_task_model import ABCTaskModel


def convert_to_saved_model(model: ABCTaskModel,
                           model_path: str,
                           version: Union[str, int] = None,
                           signatures: Any = None,
                           options: Any = None) -> None:
    """
    Export model for tensorflow serving
    Args:
        model: Target model.
        model_path: The path to which the SavedModel will be stored.
        version: The model version code, default timestamp
        signatures: Signatures to save with the SavedModel. Applicable to the
            'tf' format only. Please see the `signatures` argument in
            `tf.saved_model.save` for details.
        options: Optional `tf.saved_model.SaveOptions` object that specifies
            options for saving to SavedModel.

    """
    if not isinstance(model, ABCTaskModel):
        raise ValueError("Only supports the classification model and labeling model")
    if version is None:
        version = round(time.time())
    export_path = os.path.join(model_path, str(version))

    pathlib.Path(export_path).mkdir(exist_ok=True, parents=True)
    model.tf_model.save(export_path, save_format='tf', signatures=signatures, options=options)

    with open(os.path.join(export_path, 'model_config.json'), 'w') as f:
        f.write(json.dumps(model.to_dict(), indent=2, ensure_ascii=True))
        f.close()


if __name__ == "__main__":
    pass
