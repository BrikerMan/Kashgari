# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: seq2seq_translation.py
# time: 2:41 下午

import tensorflow as tf

# 下载文件
path_to_zip = tf.keras.utils.get_file(
    'cmn-eng.zip', origin='http://www.manythings.org/anki/cmn-eng.zip',
    extract=True)
print(path_to_zip)


if __name__ == "__main__":
    pass
