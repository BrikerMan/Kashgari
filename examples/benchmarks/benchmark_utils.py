# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: benchmark_utils.py
# time: 11:05 下午

import os
import json
from typing import List, Dict


class BenchMarkHelper:

    @classmethod
    def save_training_logs(cls,
                           log_file: str,
                           embedding_name: str,
                           model_name: str,
                           logs: List,
                           **kwargs: Dict):

        if not os.path.exists(log_file):
            data = {}
        else:
            data = json.loads(open(log_file, 'r').read())

        if embedding_name not in data:
            data[embedding_name] = {}

        data[embedding_name][model_name] = {
            'logs': logs,
            **kwargs
        }

        with open(log_file, 'w') as f:
            f.write(json.dumps(data, indent=2))


if __name__ == "__main__":
    BenchMarkHelper.save_training_logs('./training.json',
                                       embedding_name='embed_name',
                                       model_name='model_name',
                                       logs={},
                                       training_duration=321)
