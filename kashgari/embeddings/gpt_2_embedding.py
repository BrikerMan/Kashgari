# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_embedding.py
# time: 2019-05-26 17:40

import os

os.environ['TF_KERAS'] = '1'

import logging
from typing import Union, Optional, Any, List, Tuple

import numpy as np
import kashgari
import pathlib
from tensorflow.python.keras.utils import get_file
from kashgari.embeddings.base_embedding import Embedding
from kashgari.processors.base_processor import BaseProcessor
import keras_gpt_2 as gpt2


class GPT2Embedding(Embedding):
    """Pre-trained BERT embedding"""

    def info(self):
        info = super(GPT2Embedding, self).info()
        info['config'] = {
            'model_folder': self.model_folder,
            'sequence_length': self.sequence_length
        }
        return info

    def __init__(self,
                 model_folder: str,
                 task: str = None,
                 sequence_length: Union[Tuple[int, ...], str, int] = 'auto',
                 processor: Optional[BaseProcessor] = None,
                 from_saved_model: bool = False):
        """

        Args:
            task:
            model_folder:
            sequence_length:
            processor:
            from_saved_model:
        """
        super(GPT2Embedding, self).__init__(task=task,
                                            sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor,
                                            from_saved_model=from_saved_model)
        if isinstance(sequence_length, tuple):
            if len(sequence_length) > 2:
                raise ValueError('BERT only more 2')
            else:
                if not all([s == sequence_length[0] for s in sequence_length]):
                    raise ValueError('BERT only receive all')

        if sequence_length == 'variable':
            self.sequence_length = None

        self.processor.token_pad = 'pad'
        self.processor.token_unk = 'unk'
        self.processor.token_bos = 'pad'
        self.processor.token_eos = 'pad'

        self.model_folder = model_folder

        if not from_saved_model:
            self._build_token2idx_from_gpt()
            self._build_model()

    def _build_token2idx_from_gpt(self):
        encoder_path = os.path.join(self.model_folder, 'encoder.json')
        vocab_path = os.path.join(self.model_folder, 'vocab.bpe')

        bpe: gpt2.BytePairEncoding = gpt2.get_bpe_from_files(encoder_path, vocab_path)

        token2idx = bpe.token_dict.copy()
        self.processor.token2idx = token2idx
        self.processor.idx2token = dict([(value, key) for key, value in token2idx.items()])

    def _build_model(self, **kwargs):
        if self.embed_model is None and self.sequence_length != 'auto':
            config_path = os.path.join(self.model_folder, 'hparams.json')
            checkpoint_path = os.path.join(self.model_folder, 'model.ckpt')

            model = gpt2.load_trained_model_from_checkpoint(config_path,
                                                            checkpoint_path,
                                                            self.sequence_length)
            if not kashgari.config.disable_auto_summary:
                model.summary()
            self.embed_model = model
        # if self.token_count == 0:
        #     logging.debug('need to build after build_word2idx')
        # elif self.embed_model is None:
        #     seq_len = self.sequence_length
        #     if isinstance(seq_len, tuple):
        #         seq_len = seq_len[0]
        #     if isinstance(seq_len, str):
        #         return
        #     config_path = os.path.join(self.bert_path, 'bert_config.json')
        #     check_point_path = os.path.join(self.bert_path, 'bert_model.ckpt')
        #     bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
        #                                                                check_point_path,
        #                                                                seq_len=seq_len)
        #
        #     self._model = tf.keras.Model(bert_model.inputs, bert_model.output)
        #     bert_seq_len = int(bert_model.output.shape[1])
        #     if bert_seq_len < seq_len:
        #         logging.warning(f"Sequence length limit set to {bert_seq_len} by pre-trained model")
        #         self.sequence_length = bert_seq_len
        #     self.embedding_size = int(bert_model.output.shape[-1])
        #     num_layers = len(bert_model.layers)
        #     bert_model.summary()
        #     target_layer_idx = [num_layers - 1 + idx * 8 for idx in range(-3, 1)]
        #     features_layers = [bert_model.get_layer(index=idx).output for idx in target_layer_idx]
        #     embedding_layer = L.concatenate(features_layers)
        #     output_features = NonMaskingLayer()(embedding_layer)
        #
        #     self.embed_model = tf.keras.Model(bert_model.inputs, output_features)
        #     logging.warning(f'seq_len: {self.sequence_length}')

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
            self._build_token2idx_from_gpt()
        super(GPT2Embedding, self).analyze_corpus(x, y)

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

    @classmethod
    def load_data(cls, model_name):
        """
        Download pretrained GPT-2 models
        Args:
            model_name: {117M, 345M}

        Returns:
            GPT-2 model folder
        """
        model_folder: pathlib.Path = pathlib.Path(os.path.join(kashgari.macros.DATA_PATH,
                                                               'datasets',
                                                               f'gpt2-{model_name}'))
        model_folder.mkdir(exist_ok=True, parents=True)

        for filename in ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001',
                         'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:
            url = "https://storage.googleapis.com/gpt-2/models/" + model_name + "/" + filename
            get_file(os.path.join(f'gpt2-{model_name}', filename),
                     url,
                     cache_dir=kashgari.macros.DATA_PATH)
        return str(model_folder)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # bert_model_path = os.path.join(utils.get_project_path(), 'tests/test-data/bert')
    model_folder = GPT2Embedding.load_data('117M')
    print(model_folder)
    b = GPT2Embedding(task=kashgari.CLASSIFICATION,
                      model_folder=model_folder,
                      sequence_length=12)

    # from kashgari.corpus import SMP2018ECDTCorpus

    # test_x, test_y = SMP2018ECDTCorpus.load_data('valid')

    # b.analyze_corpus(test_x, test_y)
    data1 = 'all work and no play makes'.split(' ')
    r = b.embed([data1], True)
    print(r)
    print(r.shape)
