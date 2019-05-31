# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_stacked_embedding.py
# time: 2019-05-31 19:35
import os
import unittest
import numpy as np

import kashgari
from kashgari.embeddings import StackedEmbedding
from kashgari.tasks.labeling import BLSTMModel

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings import BareEmbedding, NumericFeaturesEmbedding


class TestStackedEmbedding(unittest.TestCase):

    def test_embedding(self):
        text, label = ChineseDailyNerCorpus.load_data()
        is_bold = np.random.randint(1, 3, (len(text), 12))

        text_embedding = BareEmbedding(task=kashgari.LABELING,
                                       sequence_length=12)
        num_feature_embedding = NumericFeaturesEmbedding(2,
                                                         'is_bold',
                                                         sequence_length=12)

        stack_embedding = StackedEmbedding([text_embedding, num_feature_embedding])
        stack_embedding.analyze_corpus((text, is_bold), label)

        tensor = stack_embedding.process_x_dataset((text[:3], is_bold[:3]))
        print(tensor[0].shape)
        print(tensor[1].shape)
        print(stack_embedding.embed_model.input_shape)
        print(stack_embedding.embed_model.summary())
        r = stack_embedding.embed((text[:3], is_bold[:3]))
        assert r.shape == (3, 12, 116)

    def test_training(self):
        text = ['NLP', 'Projects', 'Project', 'Name', ':']
        start_of_p = [1, 2, 1, 2, 2]
        bold = [1, 1, 1, 1, 2]
        center = [1, 1, 2, 2, 2]
        label = ['B-Category', 'I-Category', 'B-ProjectName', 'I-ProjectName', 'I-ProjectName']

        text_list = [text] * 300
        start_of_p_list = [start_of_p] * 300
        bold_list = [bold] * 300
        center_list = [center] * 300
        label_list = [label] * 300

        # You can use WordEmbedding or BERTEmbedding for your text embedding
        SEQUENCE_LEN = 100
        text_embedding = BareEmbedding(task=kashgari.LABELING, sequence_length=SEQUENCE_LEN)
        start_of_p_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                        feature_name='start_of_p',
                                                        sequence_length=SEQUENCE_LEN)

        bold_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                  feature_name='bold',
                                                  sequence_length=SEQUENCE_LEN,
                                                  embedding_size=10)

        center_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                    feature_name='center',
                                                    sequence_length=SEQUENCE_LEN)

        # first one must be the text, embedding
        stack_embedding = StackedEmbedding([
            text_embedding,
            start_of_p_embedding,
            bold_embedding,
            center_embedding
        ])

        x = (text_list, start_of_p_list, bold_list, center_list)
        y = label_list
        stack_embedding.analyze_corpus(x, y)

        model = BLSTMModel(embedding=stack_embedding)
        model.build_model(x, y)
        model.tf_model.summary()

        model.fit(x, y, epochs=2)

        model_path = os.path.join('./saved_models/',
                                  model.__class__.__module__,
                                  model.__class__.__name__)
        model.save(model_path)

        new_model = kashgari.utils.load_model(model_path)


if __name__ == "__main__":
    print("Hello world")
