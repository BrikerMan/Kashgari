# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_macros.py
# time: 3:23 下午

from kashgari.macros import DATA_PATH
from tensorflow.keras.utils import get_file


class TestMacros:
    bert_path = get_file('bert_sample_model',
                         "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                         cache_dir=DATA_PATH,
                         untar=True)

    w2v_path = get_file('sample_w2v.txt',
                        "http://s3.bmio.net/kashgari/sample_w2v.txt",
                        cache_dir=DATA_PATH)


if __name__ == "__main__":
    pass
