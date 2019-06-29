<h1 align="center">
    Kashgari
</h1>

<p align="center">
    <a href="https://github.com/BrikerMan/kashgari/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/BrikerMan/kashgari.svg?color=blue&style=popout">
    </a>
    <a href="https://travis-ci.com/BrikerMan/Kashgari">
        <img src="https://travis-ci.com/BrikerMan/Kashgari.svg?branch=tf.keras-version"/>
    </a>
    <a href='https://coveralls.io/github/BrikerMan/Kashgari?branch=tf.keras-version'>
        <img src='https://coveralls.io/repos/github/BrikerMan/Kashgari/badge.svg?branch=tf.keras-version' alt='Coverage Status'/>
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
    <a href="https://kashgari.readthedocs.io/">Contributing</a>
</h4>

## Overview

Kashgare is simple and powerful NLP framework, build your state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS) and text classification tasks.

- **Human-friendly**. Kashgare's code is straightforward, well documented and tested, which makes it very easy to understand and modify.
- **Powerful and simple**. Kashgare allows you to apply state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS) and classification.
- **Keras based**. Kashgare builds directly on Keras, making it easy to train your models and experiment with new approaches using different embeddings and model structure.
- **Buildin transfer learning**. Kashgare build-in pre-trained BERT and Word2vec embedding models, which makes it very simple to transfer learning to train your model.
- **Fully scalable**. Kashgare provide a simple, fast, and scalable environment for fast experimentation.

## Performance

| Task                     | Language | Dataset                   | Score          | Detail                                                                   |
| ------------------------ | -------- | ------------------------- | -------------- | ------------------------------------------------------------------------ |
| Named Entity Recognition | Chinese  | People's Daily Ner Corpus | **92.20** (F1) | [基于 BERT 的中文命名实体识别](https://eliyar.biz/nlp_chinese_bert_ner/) |

## Tutorials

Here is a set of quick tutorials to get you started with the library:

- [Tutorial 3: Sequence labeling Model](./tutorial/sequence_labeling_model.md)

There are also articles and posts that illustrate how to use Kashgari:

- [15 分钟搭建中文文本分类模型](https://eliyar.biz/nlp_chinese_text_classification_in_15mins/)
- [基于 BERT 的中文命名实体识别（NER)](https://eliyar.biz/nlp_chinese_bert_ner/)
- [Multi-Class Text Classification with Kashgari in 15 minutes](https://medium.com/@BrikerMan/multi-class-text-classification-with-kashgari-in-15mins-c3e744ce971d)

## Quick start

### Requirements and Installation

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

### Run with Word2vec Embedding

```python
from kashgari.embeddings import WordEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.corpus import ChineseDailyNerCorpus

bert_embedding = WordEmbedding('<Gensim embedding file>', sequence_length=30)
model = BiLSTM_CRF_Model(bert_embedding)
train_x, train_y = ChineseDailyNerCorpus.load_data()
model.fit(train_x, train_y)
```

### Support for Training on Multiple GPUs

```python
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.corpus import ChineseDailyNerCorpus

model = BiGRU_Model()
train_x, train_y = ChineseDailyNerCorpus.load_data()
model.build_multi_gpu_model(gpus=2, 
                            cpu_merge=False, 
                            cpu_relocation=False,
                            x_train=train_x, 
                            y_train=train_y)

model.fit(train_x, train_y)
```

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; start with the [contributor guidelines](CONTRIBUTING.md) and then check these open issues for specific tasks.

## Reference

This library is inspired by and references following frameworks and papers.

- [flair - A very simple framework for state-of-the-art Natural Language Processing (NLP)](https://github.com/zalandoresearch/flair)
- [anago - Bidirectional LSTM-CRF and ELMo for Named-Entity Recognition, Part-of-Speech Tagging](https://github.com/Hironsan/anago)
- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
