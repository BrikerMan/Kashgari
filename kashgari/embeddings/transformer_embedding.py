# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: transformer_embedding.py
# time: 11:41 上午

import os

os.environ['TF_KERAS'] = '1'

import json
import codecs
import logging

from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor
from bert4keras.models import build_transformer_model


class TransformerEmbedding(ABCEmbedding):

    def __init__(self,
                 vocab_path: str,
                 config_path: str,
                 checkpoint_path: str,
                 model_type: str = 'bert',
                 sequence_length: int = None,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None,
                 **kwargs):
        """
        Transformer embedding, based on https://github.com/bojone/bert4keras
        support

        Args:
            vocab_path: vocab file path, example `vocab.txt`
            config_path: model config path, example `config.json`
            checkpoint_path: model weight path, example `model.ckpt-100000`
            model_type: transfer model type, {bert, albert, nezha, gpt2_ml, t5}
            sequence_length:
            text_processor:
            label_processor:
            **kwargs:
        """
        super(TransformerEmbedding, self).__init__(sequence_length=sequence_length,
                                                   text_processor=text_processor,
                                                   label_processor=label_processor,
                                                   **kwargs)

        self.vocab_path = vocab_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type

        self.segment = True

        self.vocab_list = []
        self.max_sequence_length = None

    def build_text_vocab(self, gen: CorpusGenerator = None, force=False):
        if not self.text_processor.is_vocab_build:
            token2idx = {}
            with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
                for line in reader:
                    token = line.strip()
                    self.vocab_list.append(token)
                    token2idx[token] = len(token2idx)
            logging.debug("------ Build vocab dict finished, Top 10 token ------")
            for index, token in enumerate(self.vocab_list[:10]):
                logging.debug(f"Token: {token:8s} -> {index}")
            logging.debug("------ Build vocab dict finished, Top 10 token ------")

            self.text_processor.vocab2idx = token2idx
            self.text_processor.idx2vocab = dict([(value, key) for key, value in token2idx.items()])

    def build_embedding_model(self):
        if self.embed_model is None:
            config_path = self.config_path

            config = json.load(open(config_path))
            if 'max_position' in config:
                self.max_sequence_length = config['max_position']
            else:
                self.max_sequence_length = config.get('max_position_embeddings')

            bert_model = build_transformer_model(config_path=self.config_path,
                                                 checkpoint_path=self.checkpoint_path,
                                                 model=self.model_type,
                                                 application='encoder',
                                                 return_keras_model=True)

            self.embed_model = bert_model
            self.embedding_size = bert_model.output.shape[-1]


if __name__ == "__main__":
    vocab_path = '/Users/brikerman/Desktop/nlp/language_models/albert_base/vocab_chinese.txt'
    config_path = '/Users/brikerman/Desktop/nlp/language_models/albert_base/albert_config.json'
    checkpoint_path = '/Users/brikerman/Desktop/nlp/language_models/albert_base/model.ckpt-best'

    embed = TransformerEmbedding(vocab_path=vocab_path,
                                 config_path=config_path,
                                 checkpoint_path=checkpoint_path,
                                 model_type='albert')
    print(embed.embed(['你', '好', '啊'], debug=True).shape)
