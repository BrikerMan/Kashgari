# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_embedding.py
# time: 2019-05-20 17:40

import logging
import os
from typing import Union, Optional, Any, List, Tuple

import numpy as np
from tensorflow import keras

import kashgari.macros as k
from kashgari.embeddings.base_embedding import Embedding
from kashgari.pre_processors.base_processor import BaseProcessor

os.environ['TF_KERAS'] = '1'
import keras_bert

L = keras.layers


class BertEmbedding(Embedding):
    """Pre-trained BERT embedding"""

    def __init__(self,
                 bert_path: str,
                 task: k.TaskType = None,
                 sequence_length: Union[Tuple[int, ...], str, int] = 'auto',
                 processor: Optional[BaseProcessor] = None,
                 **kwargs):
        """

        Args:
            task:
            bert_path:
            sequence_length:
            processor:
            **kwargs:
        """
        if isinstance(sequence_length, tuple):
            if len(sequence_length) > 2:
                raise ValueError('BERT only more 2')
            else:
                if not all([s == sequence_length[0] for s in sequence_length]):
                    raise ValueError('BERT only receive all')

        if sequence_length == 'variable':
            raise ValueError('BERT only receive all')

        super(BertEmbedding, self).__init__(task=task,
                                            sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor)

        self.processor.token_pad = '[PAD]'
        self.processor.token_unk = '[UNK]'
        self.processor.token_bos = '[CLS]'
        self.processor.token_eos = '[SEP]'

        self.bert_path = bert_path
        if processor:
            self._build_token2idx_from_bert()
            self._build_model()

    def _build_token2idx_from_bert(self):
        token2idx = {
            self.processor.token_pad: 0,
            self.processor.token_unk: 1,
            self.processor.token_bos: 2,
            self.processor.token_eos: 3
        }

        dict_path = os.path.join(self.bert_path, 'vocab.txt')

        with open(dict_path, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        for idx, word in enumerate(words):
            token2idx[word] = idx
        self.bert_token2idx = token2idx
        self.processor.token2idx = self.bert_token2idx
        self.processor.idx2token = dict([(value, key) for key, value in token2idx.items()])

    def _build_model(self, **kwargs):
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            seq_len = self.sequence_length
            if isinstance(seq_len, tuple):
                seq_len = seq_len[0]
            config_path = os.path.join(self.bert_path, 'bert_config.json')
            check_point_path = os.path.join(self.bert_path, 'bert_model.ckpt')
            bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                                       check_point_path,
                                                                       seq_len=seq_len)
            self.embedding_size = bert_model.output.shape[-1]
            self.embed_model = bert_model

    def analyze_corpus(self,
                       x: Union[Tuple[List[List[str]], ...], List[List[str]]],
                       y: Union[List[List[Any]], List[Any]]):
        """
        Prepare embedding layer and pre-processor for labeling task

        Args:
            x:
            y:

        Returns:

        """
        x = utils.wrap_as_tuple(x)
        y = utils.wrap_as_tuple(y)
        if len(self.processor.token2idx) == 0:
            self._build_token2idx_from_bert()
        super(BertEmbedding, self).analyze_corpus(x, y)

    def embed(self,
              sentence_list: Union[Tuple[List[List[str]], ...], List[List[str]]]) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentence_list: Sentence list to embed

        Returns:
            vectorized sentence list
        """

        if len(sentence_list) == 1 or isinstance(sentence_list, list):
            sentence_list = (sentence_list,)
        x = self.processor.process_x_dataset(sentence_list,
                                             maxlens=self.sequence_length, )
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]

        if not isinstance(x, tuple):
            x = (x,)

        if len(x) == 1:
            segments = np.zeros(x[0].shape)
            x = (x[0], segments)
        print(x)
        embed_results = self.embed_model.predict(x)
        return embed_results


if __name__ == "__main__":
    import os
    import kashgari
    from kashgari import utils
    logging.basicConfig(level=logging.DEBUG)

    bert_path = os.path.join(utils.get_project_path(), 'tests/test-data/bert')

    b = BertEmbedding(task=kashgari.CLASSIFICATION,
                      bert_path=bert_path,
                      sequence_length=(12, 12))

    from kashgari.corpus import SMP2018ECDTCorpus

    test_x, test_y = SMP2018ECDTCorpus.load_data('valid')

    b.analyze_corpus(test_x, test_y)
    data = list('我想你啊啊啊啊啊啊啊啊')
    r = b.embed(([data], [data]))
    print(r)
    print(r.shape)
