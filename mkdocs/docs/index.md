<h1 align="center" >
    <a href='https://en.wikipedia.org/wiki/Mahmud_al-Kashgari'><strong style="color: rgba(0,0,0,.87);">Kashgari</strong></a>
</h1>

<p align="center">
    <a href="https://github.com/BrikerMan/kashgari/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/BrikerMan/kashgari.svg?color=blue&style=popout">
    </a>
    <a href="https://travis-ci.com/BrikerMan/Kashgari">
        <img src="https://travis-ci.com/BrikerMan/Kashgari.svg?branch=master"/>
    </a>
    <a href='https://coveralls.io/github/BrikerMan/Kashgari?branch=master'>
        <img src='https://coveralls.io/repos/github/BrikerMan/Kashgari/badge.svg?branch=master' alt='Coverage Status'/>
    </a>
     <a href="https://pepy.tech/project/kashgari-tf">
        <img src="https://pepy.tech/badge/kashgari-tf"/>
    </a>
    <a href="https://pypi.org/project/kashgari-tf/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/kashgari-tf.svg">
    </a>
</p>

🎉🎉🎉 We are proud to announce that we entirely rewritten Kashgari with tf.keras, now Kashgari comes with easier to understand API and is faster! 🎉🎉🎉

## Overview

Kashgari is a simple and powerful NLP Transfer learning framework, build a state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS), and text classification tasks.

- **Human-friendly**. Kashgari's code is straightforward, well documented and tested, which makes it very easy to understand and modify.
- **Powerful and simple**. Kashgari allows you to apply state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS) and classification.
- **Built-in transfer learning**. Kashgari built-in pre-trained BERT and Word2vec embedding models, which makes it very simple to transfer learning to train your model.
- **Fully scalable**. Kashgari provide a simple, fast, and scalable environment for fast experimentation, train your models and experiment with new approaches using different embeddings and model structure. 
- **Production Ready**. Kashgari could export model with `SavedModel` format for tensorflow serving, you could directly deploy it on cloud. 

## Our Goal

- **Academic users** Easier Experimentation to prove their hypothesis without coding from scratch.
- **NLP beginners** Learn how to build an NLP project with production level code quality.
- **NLP developers** Build a production level classification/labeling model within minutes.

## Performance

| Task                     | Language | Dataset                   | Score          | Detail                                                                                                             |
| ------------------------ | -------- | ------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------ |
| Named Entity Recognition | Chinese  | People's Daily Ner Corpus | **94.46** (F1) | [Text Labeling Performance Report](https://kashgari.bmio.net/tutorial/text-labeling/#performance-report) |

## Tutorials

Here is a set of quick tutorials to get you started with the library:

- [Tutorial 1: Text Classification](https://kashgari.bmio.net/tutorial/text-classification/)
- [Tutorial 2: Text Labeling](https://kashgari.bmio.net/tutorial/text-labeling/)
- [Tutorial 3: Language Embedding](https://kashgari.bmio.net/embeddings/)

There are also articles and posts that illustrate how to use Kashgari:

- [15 分钟搭建中文文本分类模型](https://eliyar.biz/nlp_chinese_text_classification_in_15mins/)
- [基于 BERT 的中文命名实体识别（NER)](https://eliyar.biz/nlp_chinese_bert_ner/)
- [BERT/ERNIE 文本分类和部署](https://eliyar.biz/nlp_train_and_deploy_bert_text_classification/)
- [五分钟搭建一个基于BERT的NER模型](https://www.jianshu.com/p/1d6689851622)
- [Multi-Class Text Classification with Kashgari in 15 minutes](https://medium.com/@BrikerMan/multi-class-text-classification-with-kashgari-in-15mins-c3e744ce971d)

## Quick start

### Requirements and Installation

🎉🎉🎉 We renamed the tf.keras version as **kashgari-tf** 🎉🎉🎉

The project is based on TenorFlow 1.14.0 and Python 3.6+, because it is 2019 and type hints is cool.

```bash
pip install kashgari-tf
# CPU
pip install tensorflow==1.14.0
# GPU
pip install tensorflow-gpu==1.14.0
```

### Example Usage

lets run a NER labeling model with Bi_LSTM Model.

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
import kashgari
from kashgari.embeddings import GPT2Embedding
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiGRU_Model

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

gpt2_embedding = GPT2Embedding('<path-to-gpt-model-folder>',
                               task=kashgari.LABELING,
                               sequence_length=30)
model = BiGRU_Model(gpt2_embedding)
model.fit(train_x, train_y, valid_x, valid_y, epochs=50)
```

### Run with Bert Embedding

```python
import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.corpus import ChineseDailyNerCorpus

bert_embedding = BERTEmbedding('<bert-model-folder>', 
                               task=kashgari.LABELING,
                               sequence_length=30)
model = BiGRU_Model(bert_embedding)

train_x, train_y = ChineseDailyNerCorpus.load_data()
model.fit(train_x, train_y)
```

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; start with the [contributor guidelines](https://kashgari.bmio.net/about/contributing/) and then check these open issues for specific tasks.

## Reference

This library is inspired by and references following frameworks and papers.

- [flair - A very simple framework for state-of-the-art Natural Language Processing (NLP)](https://github.com/zalandoresearch/flair)
- [anago - Bidirectional LSTM-CRF and ELMo for Named-Entity Recognition, Part-of-Speech Tagging](https://github.com/Hironsan/anago)
- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
