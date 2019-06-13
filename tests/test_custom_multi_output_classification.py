# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_custom_multi_output_classification.py
# time: 2019-05-22 13:36

import unittest
import numpy as np
import tensorflow as tf
import kashgari
from typing import Tuple, List, Optional, Dict, Any
from kashgari.layers import L
from kashgari.processors.classification_processor import ClassificationProcessor
from kashgari.tasks.classification.base_model import BaseClassificationModel
from kashgari.corpus import SMP2018ECDTCorpus
from tensorflow.python.keras.utils import to_categorical

train_x, train_y = SMP2018ECDTCorpus.load_data('valid')

output_1_raw = np.random.randint(3, size=len(train_x))
output_2_raw = np.random.randint(3, size=len(train_x))

output_1 = to_categorical(output_1_raw, 3)
output_2 = to_categorical(output_2_raw, 3)

print(train_x[:5])
print(output_1[:5])
print(output_2[:5])

print(len(train_x))
print(output_1.shape)
print(output_2.shape)


class MultiOutputProcessor(ClassificationProcessor):
    def process_y_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        # Data already converted to one-hot
        # Only need to get the subset
        result = []
        for index, dataset in enumerate(data):
            if subset is not None:
                target = kashgari.utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            result.append(np.array(target))

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)

    def _build_label_dict(self,
                          labels: List[str]):
        # Data already converted to one-hot
        # No need to build label dict
        self.label2idx = {1: 1, 0: 0}
        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)


class MultiOutputModel(BaseClassificationModel):
    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 256,
                'return_sequences': False
            }
        }

    def build_model_arc(self):
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']), name='layer_bi_lstm')
        layer_output_1 = L.Dense(3, activation='sigmoid', name='layer_output_1')
        layer_output_2 = L.Dense(3, activation='sigmoid', name='layer_output_2')

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor_1 = layer_output_1(tensor)
        output_tensor_2 = layer_output_2(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, [output_tensor_1, output_tensor_2])

    def predict(self,
                x_data,
                batch_size=None,
                debug_info=False,
                threshold=0.5):
        tensor = self.embedding.process_x_dataset(x_data)
        pred = self.tf_model.predict(tensor, batch_size=batch_size)

        output_1 = pred[0]
        output_2 = pred[1]

        output_1[output_1 >= threshold] = 1
        output_1[output_1 < threshold] = 0
        output_2[output_2 >= threshold] = 1
        output_2[output_2 < threshold] = 0

        return output_1, output_2


class TestCustomMultiOutputModel(unittest.TestCase):

    def test_build_and_fit(self):
        from kashgari.embeddings import BareEmbedding
        processor = MultiOutputProcessor()
        embedding = BareEmbedding(processor=processor)
        m = MultiOutputModel(embedding=embedding)
        m.build_model(train_x, (output_1, output_2))
        m.fit(train_x, (output_1, output_2), epochs=2)
        res = m.predict(train_x[:10])
        assert len(res) == 2
        assert res[0].shape == (10, 3)

    def test_build_with_BERT_and_fit(self):
        from kashgari.embeddings import BERTEmbedding
        from tensorflow.python.keras.utils import get_file
        from kashgari.macros import DATA_PATH

        sample_bert_path = get_file('bert_sample_model',
                                    "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                                    cache_dir=DATA_PATH,
                                    untar=True)

        processor = MultiOutputProcessor()
        embedding = BERTEmbedding(
            model_folder=sample_bert_path,
            processor=processor)
        m = MultiOutputModel(embedding=embedding)
        m.build_model(train_x, (output_1, output_2))
        m.fit(train_x, (output_1, output_2), epochs=2)
        res = m.predict(train_x[:10])
        assert len(res) == 2
        assert res[0].shape == (10, 3)