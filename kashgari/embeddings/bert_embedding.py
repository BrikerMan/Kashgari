# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bert_embedding.py
# time: 2:49 下午

import os
from typing import Dict, Any

from kashgari.embeddings.transformer_embedding import TransformerEmbedding


class BertEmbedding(TransformerEmbedding):
    """
    BertEmbedding is a simple wrapped class of TransformerEmbedding.
    If you need load other kind of transformer based language model, please use the TransformerEmbedding.
    """

    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(BertEmbedding, self).to_dict()
        info_dic['config']['model_folder'] = self.model_folder
        return info_dic

    def __init__(self,
                 model_folder: str,
                 **kwargs: Any):
        """

        Args:
            model_folder: path of checkpoint folder.
            kwargs: additional params
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
