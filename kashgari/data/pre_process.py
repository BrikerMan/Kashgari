# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: pre_process.py
@time: 2019-01-19 12:46

"""
import logging
import os
import pathlib
import random
import tempfile
from typing import List, Union, Dict, Tuple

import h5py
import keras
import numpy as np
import tqdm

from kashgari import macros as k
from kashgari.tokenizer import Tokenizer


class PreProcessor(object):
    def __init__(self):
        pass


def prepare_h5_file(tokenizer: Tokenizer,
                    x_data: List[List[str]],
                    y_data: Union[List[List[str]], List[int], List[str]],
                    task: k.TaskType = k.TaskType.classification,
                    step: int = 500,
                    shuffle: bool = False) -> Tuple[str, Tokenizer]:
    assert len(x_data) == len(y_data)
    if shuffle:
        x_data, y_data = unison_shuffled_copies(x_data, y_data)

    x_data = x_data
    y_data = y_data
    label2idx = generate_label2idx_dict(y_data)
    tokenizer.label2idx = label2idx
    tokenizer.idx2label = dict([(value, key) for key, value in tokenizer.label2idx.items()])

    tmp_folder = tempfile.TemporaryDirectory().name
    pathlib.Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    h5_path = os.path.join(tmp_folder, 'dataset.h5')
    h5 = h5py.File(h5_path, 'a')

    try:
        h5.create_dataset('x',
                          shape=(5, tokenizer.sequence_length),
                          maxshape=(None, tokenizer.sequence_length),
                          dtype=np.int32,
                          chunks=True)
    except Exception:
        pass

    page = len(x_data) // step
    if len(x_data) % step != 0:
        page += 1

    for page_index in tqdm.tqdm(range(page), desc='processing input sequence'):
        start_index = page * page_index
        x_padded = tokenize_text_sequence(tokenizer, x_data[start_index: start_index + step])
        new_count = start_index + len(x_padded)
        if new_count > h5['x'].shape[0]:
            h5['x'].resize((new_count, tokenizer.sequence_length))
        h5['x'][start_index:new_count] = x_padded

    if task == k.TaskType.classification:
        padding_y = [tokenizer.label2idx[y] for y in y_data]
        padding_y = np.array(padding_y)
        h5.create_dataset('y', data=padding_y)
    h5.close()
    logging.info('processed data saved to: {}'.format(h5_path))
    return tmp_folder, tokenizer


def generate_label2idx_dict(y_data: Union[List[List[str]], List[int], List[str]]) -> Dict[str, int]:
    label_set = []
    for y_item in y_data:
        if isinstance(y_item, str):
            label_set.append(y_item)
        elif isinstance(y_item, int):
            label_set.append(y_item)
        else:
            for item in y_item:
                label_set.append(item)
    label_set = set(label_set)
    label2idx = {}
    for label in label_set:
        label2idx[label] = len(label2idx)
    return label2idx


def tokenize_text_sequence(tokenizer: Tokenizer,
                           text_sequences: List[List[str]]):
    tokenized_seq = [tokenizer.word_to_token(text_seq) for text_seq in text_sequences]
    seq = keras.preprocessing.sequence.pad_sequences(tokenized_seq,
                                                     maxlen=tokenizer.sequence_length,
                                                     padding='post')
    return seq


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


if __name__ == "__main__":
    # df = pd.read_csv('/Users/brikerman/Downloads/simplifyweibo_4_moods.csv')
    # df = pre_process_df(df)
    # df.to_csv('dataset.csv')
    pass
