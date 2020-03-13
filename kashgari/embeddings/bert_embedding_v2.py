# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bert_embedding_v2.py
# time: 10:03 上午

import os

os.environ['TF_KERAS'] = '1'

import json
import codecs
import logging
from typing import Union, Optional
from bert4keras.models import build_transformer_model
import kashgari
import tensorflow as tf
from kashgari.embeddings.bert_embedding import BERTEmbedding
from kashgari.layers import NonMaskingLayer
from kashgari.processors.base_processor import BaseProcessor
import keras_bert


class BERTEmbeddingV2(BERTEmbedding):
    """Pre-trained BERT embedding"""

    def info(self):
        info = super(BERTEmbedding, self).info()
        info['config'] = {
            'model_folder': self.model_folder,
            'sequence_length': self.sequence_length
        }
        return info

    def __init__(self,
                 vacab_path: str,
                 config_path: str,
                 checkpoint_path: str,
                 bert_type: str = 'bert',
                 task: str = None,
                 sequence_length: Union[str, int] = 'auto',
                 processor: Optional[BaseProcessor] = None,
                 from_saved_model: bool = False):
        """
        """
        self.model_folder = ''
        self.vacab_path = vacab_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        super(BERTEmbedding, self).__init__(task=task,
                                            sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor,
                                            from_saved_model=from_saved_model)
        self.bert_type = bert_type
        self.processor.token_pad = '[PAD]'
        self.processor.token_unk = '[UNK]'
        self.processor.token_bos = '[CLS]'
        self.processor.token_eos = '[SEP]'

        self.processor.add_bos_eos = True

        if not from_saved_model:
            self._build_token2idx_from_bert()
            self._build_model()

    def _build_token2idx_from_bert(self):
        token2idx = {}
        with codecs.open(self.vacab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token2idx[token] = len(token2idx)

        self.bert_token2idx = token2idx
        self._tokenizer = keras_bert.Tokenizer(token2idx)
        self.processor.token2idx = self.bert_token2idx
        self.processor.idx2token = dict([(value, key) for key, value in token2idx.items()])

    def _build_model(self, **kwargs):
        if self.embed_model is None:
            seq_len = self.sequence_length
            if isinstance(seq_len, tuple):
                seq_len = seq_len[0]
            if isinstance(seq_len, str):
                logging.warning(f"Model will be built when sequence length is determined")
                return

            config_path = self.config_path

            config = json.load(open(config_path))
            if seq_len > config.get('max_position_embeddings'):
                seq_len = config.get('max_position_embeddings')
                logging.warning(f"Max seq length is {seq_len}")

            bert_model = build_transformer_model(config_path=self.config_path,
                                                 checkpoint_path=self.checkpoint_path,
                                                 model=self.bert_type,
                                                 application='encoder',
                                                 return_keras_model=True)

            self.embed_model = bert_model

            self.embedding_size = int(bert_model.output.shape[-1])
            output_features = NonMaskingLayer()(bert_model.output)
            self.embed_model = tf.keras.Model(bert_model.inputs, output_features)


if __name__ == "__main__":
    # BERT_PATH = '/Users/brikerman/Desktop/nlp/language_models/bert/chinese_L-12_H-768_A-12'
    model_folder = '/Users/brikerman/Desktop/nlp/language_models/albert_base'
    checkpoint_path = os.path.join(model_folder, 'model.ckpt-best')
    config_path = os.path.join(model_folder, 'albert_config.json')
    vacab_path = os.path.join(model_folder, 'vocab_chinese.txt')
    embed = BERTEmbeddingV2(vacab_path, config_path, checkpoint_path,
                            bert_type='albert',
                            task=kashgari.CLASSIFICATION,
                            sequence_length=100)
    x = embed.embed_one(list('今天天气不错'))
    print(x)
