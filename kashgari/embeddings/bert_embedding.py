# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: bert_embedding.py
# time: 2:49 下午

import os
from kashgari.processors.abc_processor import ABCProcessor
from kashgari.embeddings.transformer_embedding import TransformerEmbedding


class BertEmbedding(TransformerEmbedding):
    def __init__(self,
                 model_folder: str,
                 layer_nums: int = 4,
                 sequence_length: int = None,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None,
                 **kwargs):
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
        check_point_path = os.path.join(self.model_folder, 'bert_model.ckpt')
        super(BertEmbedding, self).__init__(vocab_path=vocab_path,
                                            config_path=config_path,
                                            checkpoint_path=check_point_path,
                                            model_type='bert',
                                            layer_nums=layer_nums,
                                            sequence_length=sequence_length,
                                            text_processor=text_processor,
                                            label_processor=label_processor,
                                            **kwargs)


if __name__ == "__main__":
    pass
