# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_embedding.py
# time: 2019-05-25 17:40

import os

os.environ['TF_KERAS'] = '1'

import codecs
import logging
from typing import Union, Optional, Any, List, Tuple

import numpy as np
import kashgari
import tensorflow as tf
from kashgari.layers import NonMaskingLayer
from kashgari.embeddings.base_embedding import Embedding
from kashgari.processors.base_processor import BaseProcessor
import keras_bert


class BERTEmbedding(Embedding):
    """Pre-trained BERT embedding"""

    def info(self):
        info = super(BERTEmbedding, self).info()
        info['config'] = {
            'model_folder': self.model_folder,
            'sequence_length': self.sequence_length
        }
        return info

    def __init__(self,
                 model_folder: str,
                 layer_nums: int = 4,
                 trainable: bool = False,
                 task: str = None,
                 sequence_length: Union[str, int] = 'auto',
                 processor: Optional[BaseProcessor] = None,
                 from_saved_model: bool = False):
        """

        Args:
            task:
            model_folder:
            layer_nums: number of layers whose outputs will be concatenated into a single tensor,
                           default `4`, output the last 4 hidden layers as the thesis suggested
            trainable: whether if the model is trainable, default `False` and set it to `True`
                        for fine-tune this embedding layer during your training
            sequence_length:
            processor:
            from_saved_model:
        """
        self.trainable = trainable
        # Do not need to train the whole bert model if just to use its feature output
        self.training = False
        self.layer_nums = layer_nums
        if isinstance(sequence_length, tuple):
            raise ValueError('BERT embedding only accept `int` type `sequence_length`')

        if sequence_length == 'variable':
            raise ValueError('BERT embedding only accept sequences in equal length')

        super(BERTEmbedding, self).__init__(task=task,
                                            sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor,
                                            from_saved_model=from_saved_model)

        self.processor.token_pad = '[PAD]'
        self.processor.token_unk = '[UNK]'
        self.processor.token_bos = '[CLS]'
        self.processor.token_eos = '[SEP]'

        self.processor.add_bos_eos = True

        self.model_folder = model_folder
        if not from_saved_model:
            self._build_token2idx_from_bert()
            self._build_model()

    def _build_token2idx_from_bert(self):
        dict_path = os.path.join(self.model_folder, 'vocab.txt')

        token2idx = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
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
                logging.warning(f"Model will be built until sequence length is determined")
                return
            config_path = os.path.join(self.model_folder, 'bert_config.json')
            check_point_path = os.path.join(self.model_folder, 'bert_model.ckpt')
            bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                                       check_point_path,
                                                                       seq_len=seq_len,
                                                                       output_layer_num=self.layer_nums,
                                                                       training=self.training,
                                                                       trainable=self.trainable)

            self._model = tf.keras.Model(bert_model.inputs, bert_model.output)
            bert_seq_len = int(bert_model.output.shape[1])
            if bert_seq_len < seq_len:
                logging.warning(f"Sequence length limit set to {bert_seq_len} by pre-trained model")
                self.sequence_length = bert_seq_len
            self.embedding_size = int(bert_model.output.shape[-1])
            output_features = NonMaskingLayer()(bert_model.output)
            self.embed_model = tf.keras.Model(bert_model.inputs, output_features)
            logging.warning(f'seq_len: {self.sequence_length}')

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
        if len(self.processor.token2idx) == 0:
            self._build_token2idx_from_bert()
        super(BERTEmbedding, self).analyze_corpus(x, y)

    def embed(self,
              sentence_list: Union[Tuple[List[List[str]], ...], List[List[str]]],
              debug: bool = False) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentence_list: Sentence list to embed
            debug: show debug log
        Returns:
            vectorized sentence list
        """
        if self.embed_model is None:
            raise ValueError('need to build model for embed sentence')

        tensor_x = self.process_x_dataset(sentence_list)
        if debug:
            logging.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results

    def process_x_dataset(self,
                          data: Union[Tuple[List[List[str]], ...], List[List[str]]],
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        """
        batch process feature data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        x1 = None
        if isinstance(data, tuple):
            if len(data) == 2:
                x0 = self.processor.process_x_dataset(data[0], self.sequence_length, subset)
                x1 = self.processor.process_x_dataset(data[1], self.sequence_length, subset)
            else:
                x0 = self.processor.process_x_dataset(data[0], self.sequence_length, subset)
        else:
            x0 = self.processor.process_x_dataset(data, self.sequence_length, subset)
        if x1 is None:
            x1 = np.zeros(x0.shape, dtype=np.int32)
        return x0, x1


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # bert_model_path = os.path.join(utils.get_project_path(), 'tests/test-data/bert')

    b = BERTEmbedding(task=kashgari.CLASSIFICATION,
                      model_folder='/Users/brikerman/.kashgari/embedding/bert/chinese_L-12_H-768_A-12',
                      sequence_length=12)

    from kashgari.corpus import SMP2018ECDTCorpus

    test_x, test_y = SMP2018ECDTCorpus.load_data('valid')

    b.analyze_corpus(test_x, test_y)
    data1 = 'all work and no play makes'.split(' ')
    data2 = '你 好 啊'.split(' ')
    r = b.embed([data1], True)

    tokens = b.process_x_dataset([['语', '言', '模', '型']])[0]
    target_index = [101, 6427, 6241, 3563, 1798, 102]
    target_index = target_index + [0] * (12 - len(target_index))
    assert list(tokens[0]) == list(target_index)
    print(tokens)
    print(r)
    print(r.shape)
