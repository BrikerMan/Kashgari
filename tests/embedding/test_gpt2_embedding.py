# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_gpt2_embedding.py
# time: 2019-05-31 19:33

import kashgari
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.embeddings import GPT2Embedding
from kashgari.processors import LabelingProcessor

import tests.embedding.test_bare_embedding as base


class TestGPT2Embedding(base.TestBareEmbedding):
    @classmethod
    def setUpClass(cls):
        cls.embedding_class = GPT2Embedding
        cls.config = {
            'model_folder': GPT2Embedding.load_data('117M')
        }

    def test_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         **self.config)

        valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (15, 50257)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 15, 50257)

        embedding = self.embedding_class(task=kashgari.LABELING,
                                         sequence_length=10,
                                         **self.config)

        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        embedding.analyze_corpus(valid_x, valid_y)

        assert embedding.embed_one(['我', '想', '看']).shape == (10, 50257)

        assert embedding.embed([
            ['我', '想', '看'],
            ['我', '想', '看', '权力的游戏'],
            ['Hello', 'world']
        ]).shape == (3, 10, 50257)

    def test_variable_length_embed(self):
        embedding = self.embedding_class(task=kashgari.CLASSIFICATION,
                                         sequence_length='variable',
                                         **self.config)

        assert embedding.embed([
            ['Hello', 'world']
        ]).shape == (1, 2, 50257)

        assert embedding.embed([
            ['Hello', 'world', 'kashgari']
        ]).shape == (1, 3, 50257)

    def test_init_with_processor(self):
        valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

        processor = LabelingProcessor()
        processor.analyze_corpus(valid_x, valid_y)

        embedding = self.embedding_class(sequence_length=11,
                                         processor=processor,
                                         **self.config)
        embedding.analyze_corpus(valid_x, valid_y)
        assert embedding.embed_one(['我', '想', '看']).shape == (11, 50257)


if __name__ == "__main__":
    print("Hello world")
