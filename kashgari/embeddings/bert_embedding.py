# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bert_embedding.py
# time: 2:49 下午

import os
from typing import Dict, Any

from kashgari.embeddings.transformer_embedding import TransformerEmbedding
from kashgari.processors.abc_processor import ABCProcessor


class BertEmbedding(TransformerEmbedding):
    def info(self) -> Dict:
        info_dic = super(BertEmbedding, self).info()
        info_dic['config']['model_folder'] = self.model_folder
        return info_dic

    def __init__(self,
                 model_folder: str,
                 **kwargs: Any) -> None:
        """

        Args:
            model_folder: path of checkpoint folder.
            sequence_length: If using an integer, let's say 50, the input output sequence length will set to 50.
                If not set will use the 95% of corpus length as sequence length.
            text_processor:
            label_processor:

            **kwargs:
        """
        self.model_folder = model_folder
        vocab_path = os.path.join(self.model_folder, 'vocab.txt')
        config_path = os.path.join(self.model_folder, 'bert_config.json')
        checkpoint_path = os.path.join(self.model_folder, 'bert_model.ckpt')
        kwargs['vocab_path'] = vocab_path
        kwargs['config_path'] = config_path
        kwargs['checkpoint_path'] = checkpoint_path
        kwargs['model_type'] = 'bert'
        super(BertEmbedding, self).__init__(**kwargs)


if __name__ == "__main__":
    pass
