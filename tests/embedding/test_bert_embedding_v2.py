# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_bert_embedding.py
# time: 2019-05-31 19:32

import os
import pytest
import unittest
import kashgari
from kashgari.embeddings.bert_embedding_v2 import BERTEmbeddingV2
from kashgari.tokenizer import BertTokenizer

@pytest.mark.skip
class TestBERTEmbedding(unittest.TestCase):

    def test_basic_use(self):
        model_folder = '/Users/brikerman/Desktop/nlp/language_models/albert_base'

        checkpoint_path = os.path.join(model_folder, 'model.ckpt-best')
        config_path = os.path.join(model_folder, 'albert_config.json')
        vacab_path = os.path.join(model_folder, 'vocab_chinese.txt')

        tokenizer = BertTokenizer.load_from_vacab_file(vacab_path)
        embed = BERTEmbeddingV2(vacab_path, config_path, checkpoint_path,
                                bert_type='albert',
                                task=kashgari.CLASSIFICATION,
                                sequence_length=100)

        sentences = [
            "Jim Henson was a puppeteer.",
            "This here's an example of using the BERT tokenizer.",
            "Why did the chicken cross the road?"
        ]
        labels = [
            "class1",
            "class2",
            "class1"
        ]

        sentences_tokenized = [tokenizer.tokenize(s) for s in sentences]
        print(sentences_tokenized)

        train_x, train_y = sentences_tokenized[:2], labels[:2]
        validate_x, validate_y = sentences_tokenized[2:], labels[2:]

        from kashgari.tasks.classification import CNNLSTMModel
        model = CNNLSTMModel(embed)

        # ------------ build model ------------
        model.fit(
            train_x, train_y,
            validate_x, validate_y,
            epochs=3,
            batch_size=32
        )


if __name__ == "__main__":
    print("Hello world")
