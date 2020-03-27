<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<h1 align="center">
    <a href='https://en.wikipedia.org/wiki/Mahmud_al-Kashgari'>Kashgari</a>
</h1>

<p align="center">
    <a href="https://github.com/BrikerMan/kashgari/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/BrikerMan/kashgari.svg?color=blue&style=popout">
    </a>
    <a href="https://join.slack.com/t/kashgari/shared_invite/enQtODU4OTEzNDExNjUyLTY0MzI4MGFkZmRkY2VmMzdmZjRkZTYxMmMwNjMyOTI1NGE5YzQ2OTZkYzA1YWY0NTkyMDdlZGY5MGI5N2U4YzM">
        <img alt="Slack" src="https://img.shields.io/badge/chat-Slack-blueviolet?logo=Slack&style=popout">
    </a>
    <a href="https://travis-ci.com/BrikerMan/Kashgari">
        <img src="https://travis-ci.com/BrikerMan/Kashgari.svg?branch=master"/>
    </a>
    <a href='https://coveralls.io/github/BrikerMan/Kashgari?branch=master'>
        <img src='https://coveralls.io/repos/github/BrikerMan/Kashgari/badge.svg?branch=master' alt='Coverage Status'/>
    </a>
     <a href="https://pepy.tech/project/kashgari">
        <img src="https://pepy.tech/badge/kashgari"/>
    </a>
    <a href="https://pypi.org/project/kashgari/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/kashgari.svg">
    </a>
</p>

<h4 align="center">
    <a href="#overview">Overview</a> |
    <a href="#performance">Performance</a> |
    <a href="#quick-start">Quick start</a> |
    <a href="https://kashgari.readthedocs.io/">Documentation</a> |
    <a href="https://kashgari.readthedocs.io/about/contributing/">Contributing</a>
</h4>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->

🎉🎉🎉 We are proud to announce that we entirely rewrote Kashgari with tf.keras, now Kashgari comes with easier to understand API and is faster! 🎉🎉🎉

## Overview

Kashgari is a simple and powerful NLP Transfer learning framework, build a state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS), and text classification tasks.

- **Human-friendly**. Kashgari's code is straightforward, well documented and tested, which makes it very easy to understand and modify.
- **Powerful and simple**. Kashgari allows you to apply state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS) and classification.
- **Built-in transfer learning**. Kashgari built-in pre-trained BERT and Word2vec embedding models, which makes it very simple to transfer learning to train your model.
- **Fully scalable**. Kashgari provides a simple, fast, and scalable environment for fast experimentation, train your models and experiment with new approaches using different embeddings and model structure.
- **Production Ready**. Kashgari could export model with `SavedModel` format for tensorflow serving, you could directly deploy it on the cloud.

## Our Goal

- **Academic users** Easier experimentation to prove their hypothesis without coding from scratch.
- **NLP beginners** Learn how to build an NLP project with production level code quality.
- **NLP developers** Build a production level classification/labeling model within minutes.

## Performance

| Task                     | Language | Dataset                   | Score          | Detail                                                                                                   |
| ------------------------ | -------- | ------------------------- | -------------- | -------------------------------------------------------------------------------------------------------- |
| Named Entity Recognition | Chinese  | People's Daily Ner Corpus | **94.46** (F1) | [Text Labeling Performance Report](https://kashgari.rtfd.io/tutorial/text-labeling.html#performance-report) |

## Tutorials

Here is a set of quick tutorials to get you started with the library:

- [Tutorial 1: Text Classification](./docs/tutorial/text-classification.md)
- [Tutorial 2: Text Labeling](./docs/tutorial/text-labeling.md)
- [Tutorial 3: Text Scoring](./docs/tutorial/text-scoring.md)
- [Tutorial 4: Language Embedding](./docs/embeddings/index.md)

There are also articles and posts that illustrate how to use Kashgari:

- [15 分钟搭建中文文本分类模型](https://eliyar.biz/nlp_chinese_text_classification_in_15mins/)
- [基于 BERT 的中文命名实体识别（NER)](https://eliyar.biz/nlp_chinese_bert_ner/)
- [BERT/ERNIE 文本分类和部署](https://eliyar.biz/nlp_train_and_deploy_bert_text_classification/)
- [五分钟搭建一个基于BERT的NER模型](https://www.jianshu.com/p/1d6689851622)
- [Multi-Class Text Classification with Kashgari in 15 minutes](https://medium.com/@BrikerMan/multi-class-text-classification-with-kashgari-in-15mins-c3e744ce971d)

## Quick start

### Requirements and Installation

🎉🎉🎉 We renamed again for consistency and clarity. From now on, it is all `kashgari`. 🎉🎉🎉

The project is based on Python 3.6+, because it is 2019 and type hinting is cool.

| Backend          | pypi version                           | desc            |
| ---------------- | -------------------------------------- | --------------- |
| TensorFlow 2.x   | `pip install 'kashgari>=2.0.0'`        | coming soon     |
| TensorFlow 1.14+ | `pip install 'kashgari>=1.0.0,<2.0.0'` | current version |
| Keras            | `pip install 'kashgari<1.0.0'`         | legacy version  |

[Find more info about the name changing.](https://github.com/BrikerMan/Kashgari/releases/tag/v1.0.0)

### Example Usage

Let's run an NER labeling model with Bi\_LSTM Model.

```python
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_Model

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

model = BiLSTM_Model()
model.fit(train_x, train_y, valid_x, valid_y, epochs=50)

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 97)                0
_________________________________________________________________
layer_embedding (Embedding)  (None, 97, 100)           320600
_________________________________________________________________
layer_blstm (Bidirectional)  (None, 97, 256)           235520
_________________________________________________________________
layer_dropout (Dropout)      (None, 97, 256)           0
_________________________________________________________________
layer_time_distributed (Time (None, 97, 8)             2056
_________________________________________________________________
activation_7 (Activation)    (None, 97, 8)             0
=================================================================
Total params: 558,176
Trainable params: 558,176
Non-trainable params: 0
_________________________________________________________________
Train on 20864 samples, validate on 2318 samples
Epoch 1/50
20864/20864 [==============================] - 9s 417us/sample - loss: 0.2508 - acc: 0.9333 - val_loss: 0.1240 - val_acc: 0.9607

"""
```

### Run with GPT-2 Embedding

```python
from kashgari.embeddings import GPT2Embedding
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiGRU_Model

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

gpt2_embedding = GPT2Embedding('<path-to-gpt-model-folder>', sequence_length=30)
model = BiGRU_Model(gpt2_embedding)
model.fit(train_x, train_y, valid_x, valid_y, epochs=50)
```

### Run with Bert Embedding

```python
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.corpus import ChineseDailyNerCorpus

bert_embedding = BERTEmbedding('<bert-model-folder>', sequence_length=30)
model = BiGRU_Model(bert_embedding)

train_x, train_y = ChineseDailyNerCorpus.load_data()
model.fit(train_x, train_y)
```

## Sponsors

Support this project by becoming a sponsor. Your issues and feature request will be prioritized.[[Become a sponsor](https://www.patreon.com/join/brikerman?)]

## Contributors ✨

Thanks goes to these wonderful people. And there are many ways to get involved. Start with the [contributor guidelines](./docs/about/contributing.md) and then check these open issues for specific tasks.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://developers.google.com/community/experts/directory/profile/profile-eliyar_eziz"><img src="https://avatars1.githubusercontent.com/u/9368907?v=4" width="100px;" alt=""/><br /><sub><b>Eliyar Eziz</b></sub></a><br /><a href="https://github.com/BrikerMan/Kashgari/commits?author=BrikerMan" title="Documentation">📖</a> <a href="https://github.com/BrikerMan/Kashgari/commits?author=BrikerMan" title="Tests">⚠️</a> <a href="https://github.com/BrikerMan/Kashgari/commits?author=BrikerMan" title="Code">💻</a></td>
    <td align="center"><a href="http://www.chuanxilu.com"><img src="https://avatars3.githubusercontent.com/u/856746?v=4" width="100px;" alt=""/><br /><sub><b>Alex Wang</b></sub></a><br /><a href="https://github.com/BrikerMan/Kashgari/commits?author=alexwwang" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/lsgrep"><img src="https://avatars3.githubusercontent.com/u/3893940?v=4" width="100px;" alt=""/><br /><sub><b>Yusup</b></sub></a><br /><a href="https://github.com/BrikerMan/Kashgari/commits?author=lsgrep" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/adlinex"><img src="https://avatars1.githubusercontent.com/u/5442229?v=4" width="100px;" alt=""/><br /><sub><b>Adline</b></sub></a><br /><a href="https://github.com/BrikerMan/Kashgari/commits?author=adlinex" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

Feel free to join the Slack group if you want to more involved in Kashgari's development.

[Slack Group Link](https://join.slack.com/t/kashgari/shared_invite/enQtODU4OTEzNDExNjUyLTY0MzI4MGFkZmRkY2VmMzdmZjRkZTYxMmMwNjMyOTI1NGE5YzQ2OTZkYzA1YWY0NTkyMDdlZGY5MGI5N2U4YzM)

## Reference

This library is inspired by and references following frameworks and papers.

- [flair - A very simple framework for state-of-the-art Natural Language Processing (NLP)](https://github.com/zalandoresearch/flair)
- [anago - Bidirectional LSTM-CRF and ELMo for Named-Entity Recognition, Part-of-Speech Tagging](https://github.com/Hironsan/anago)
- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- [bert4keras - Our light reimplement of bert for keras](https://github.com/bojone/bert4keras/)
