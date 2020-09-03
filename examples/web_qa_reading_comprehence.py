#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/2 5:51 下午
# File    : web_qa_reading_comprehence.py
# Project : Kashgari

import os
import json
from kashgari.tokenizers.bert_tokenizer import BertTokenizer
from kashgari.embeddings import BertEmbedding
from examples.tools import get_bert_path
from kashgari.tasks.seq2seq.model import Seq2Seq
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

WEB_QA_PATH = '/home/brikerman/Downloads/SogouQA.json'
Sogou_QA_PATH = '/home/brikerman/Downloads/SogouQA.json'

with open(Sogou_QA_PATH, 'r') as f:
    corpus_data = json.loads(f.read())

bert_path = get_bert_path()
tokenizer = BertTokenizer.load_from_vocab_file(os.path.join(bert_path, 'vocab.txt'))

# 筛选数据
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
x_data = []
y_data = []

for d in corpus_data:
    for p in d['passages']:
        if p['answer']:
            x = tokenizer.tokenize(d['question']) + ['[SEP]'] + tokenizer.tokenize(p['passage'])
            x_data.append(x)
            y_data.append(tokenizer.tokenize(p['answer']))

print(x_data[:3])
print(y_data[:3])

bert = BertEmbedding(bert_path)
model = Seq2Seq(encoder_seq_length=256)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.sample_count = 5

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 != 0:
            return
        import random
        samples = random.sample(x_data, self.sample_count)
        translates, _ = self.model.predict(samples)
        print()
        for index in range(len(samples)):
            print(f"X: {''.join(samples[index])}")
            print(f"Y: {''.join(translates[index])}")
            print('------------------------------')


his_callback = CustomCallback(model)
history = model.fit(x_data,
                    y_data,
                    callbacks=[his_callback],
                    epochs=50,
                    batch_size=16)

if __name__ == '__main__':
    pass
