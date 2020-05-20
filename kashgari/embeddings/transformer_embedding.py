# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: transformer_embedding.py
# time: 11:41 上午

import json
import codecs
import logging
from typing import Dict, List, Any, Optional
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.generators import CorpusGenerator
from kashgari.processors.abc_processor import ABCProcessor
from bert4keras.models import build_transformer_model


class TransformerEmbedding(ABCEmbedding):
    def info(self) -> Dict:
        info_dic = super(TransformerEmbedding, self).info()
        info_dic['config']['vocab_path'] = self.vocab_path
        info_dic['config']['config_path'] = self.config_path
        info_dic['config']['checkpoint_path'] = self.checkpoint_path
        info_dic['config']['model_type'] = self.model_type
        return info_dic

    def __init__(self,
                 vocab_path: str,
                 config_path: str,
                 checkpoint_path: str,
                 model_type: str = 'bert',
                 **kwargs: Any) -> None:
        """
        Transformer embedding, based on https://github.com/bojone/bert4keras
        support

        Args:
            vocab_path: vocab file path, example `vocab.txt`
            config_path: model config path, example `config.json`
            checkpoint_path: model weight path, example `model.ckpt-100000`
            model_type: transfer model type, {bert, albert, nezha, gpt2_ml, t5}
            layer_nums: number of layers whose outputs will be concatenated into a single tensor, default 1,
                output the last 1 hidden layers.
            sequence_length:
            text_processor:
            label_processor:
            **kwargs:
        """
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.segment = True
        self.vocab_list: List[str] = []

        super(TransformerEmbedding, self).__init__(**kwargs)

    def load_embed_vocab(self) -> Optional[Dict[str, int]]:
        token2idx: Dict[str, int] = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.vocab_list.append(token)
                token2idx[token] = len(token2idx)
        logging.debug("------ Build vocab dict finished, Top 10 token ------")
        for index, token in enumerate(self.vocab_list[:10]):
            logging.debug(f"Token: {token:8s} -> {index}")
        logging.debug("------ Build vocab dict finished, Top 10 token ------")
        return token2idx

    def build_embedding_model(self,
                              *,
                              vocab_size: int = None,
                              force: bool = False,
                              **kwargs: Dict) -> None:
        if self.embed_model is None:
            config_path = self.config_path
            config = json.loads(open(config_path, 'r').read())
            if 'max_position' in config:
                self.max_position = config['max_position']
            else:
                self.max_position = config.get('max_position_embeddings')

            bert_model = build_transformer_model(config_path=self.config_path,
                                                 checkpoint_path=self.checkpoint_path,
                                                 model=self.model_type,
                                                 application='encoder',
                                                 return_keras_model=True)
            for layer in bert_model.layers:
                layer.trainable = False
            self.embed_model = bert_model
            self.embedding_size = bert_model.output.shape[-1]


if __name__ == "__main__":
    pass
    # vocab_path = '/Users/brikerman/Desktop/nlp/language_models/albert_base/vocab_chinese.txt'
    # config_path = '/Users/brikerman/Desktop/nlp/language_models/albert_base/albert_config.json'
    # checkpoint_path = '/Users/brikerman/Desktop/nlp/language_models/albert_base/model.ckpt-best'
    #
    # embed = TransformerEmbedding(vocab_path=vocab_path,
    #                              config_path=config_path,
    #                              checkpoint_path=checkpoint_path,
    #                              model_type='albert')
    # print(embed.embed(['你', '好', '啊'], debug=True).shape)
