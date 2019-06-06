# Kashgari

[![Pypi](https://img.shields.io/pypi/v/kashgari.svg)](https://pypi.org/project/kashgari/)
[![Python version](https://img.shields.io/pypi/pyversions/Kashgari.svg)](https://www.python.org/downloads/release/python-360/)
![Travis (.com) branch](https://img.shields.io/travis/com/BrikerMan/Kashgari/master.svg)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FBrikerMan%2FKashgari.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FBrikerMan%2FKashgari?ref=badge_shield)
[![](https://img.shields.io/coveralls/github/BrikerMan/Kashgari.svg)](https://coveralls.io/github/BrikerMan/Kashgari)
[![Issues](https://img.shields.io/github/issues/BrikerMan/Kashgari.svg)](https://github.com/BrikerMan/Kashgari/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
![](https://img.shields.io/pypi/l/kashgari.svg?style=flat)
[![](https://img.shields.io/pypi/dm/kashgari.svg)](https://pypi.org/project/kashgari/)

Simple and powerful NLP framework, build your state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS) and text classification tasks.

Kashgare is:

* **Human-friendly**. Kashgare's code is straightforward, well documented and tested, which makes it very easy to understand and modify.
* **Powerful and simple**. Kashgare allows you to apply state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS) and classification.
* **Keras based**. Kashgare builds directly on Keras, making it easy to train your models and experiment with new approaches using different embeddings and model structure.
* **Easy to fine-tune**. Kashgare build-in pre-trained BERT and Word2vec embedding models, which makes it very simple to fine-tune your model based on this embeddings.
* **Fully scalable**. Kashgare provide a simple, fast, and scalable environment for fast experimentation.
 
## Feature List 

* Embedding support
    * Classic word2vec embedding
    * BERT embedding
    * GPT-2 embedding
* Sequence(Text) Classification Models
    * CNNModel
    * BLSTMModel
    * CNNLSTMModel
    * AVCNNModel
    * KMaxCNNModel
    * RCNNModel
    * AVRNNModel
    * DropoutBGRUModel
    * DropoutAVRNNModel
* Sequence(Text) Labeling Models (NER, PoS)
    * CNNLSTMModel
    * BLSTMModel
    * BLSTMCRFModel
* Model Training
* Model Evaluate
* GPU Support / Multi GPU Support
* Customize Model

## Performance

| Task                     | Language | Dataset                   | Score          | Detail                                                                   |
| ------------------------ | -------- | ------------------------- | -------------- | ------------------------------------------------------------------------ |
| Named Entity Recognition | Chinese  | People's Daily Ner Corpus | **92.20** (F1) | [基于 BERT 的中文命名实体识别](https://eliyar.biz/nlp_chinese_bert_ner/) |

## Roadmap

* [ ] **[Migrate to tf.keras](https://github.com/BrikerMan/Kashgari/issues/77)**
* [ ] ELMo Embedding
* [ ] Pre-trained models
* [ ] More model structure

## Tutorials

Here is a set of quick tutorials to get you started with the library:

* [Tutorial 1: Word Embeddings](docs/Tutorial_1_Embedding.md)
* [Tutorial 2: Classification Model](docs/Tutorial_2_Classification.md)
* [Tutorial 3: Sequence labeling Model](docs/Tutorial_3_Sequence_Labeling.md)

There are also articles and posts that illustrate how to use Kashgari:

* [15分钟搭建中文文本分类模型](https://eliyar.biz/nlp_chinese_text_classification_in_15mins/)
* [基于 BERT 的中文命名实体识别（NER)](https://eliyar.biz/nlp_chinese_bert_ner/)
* [Multi-Class Text Classification with Kashgari in 15 minutes](https://medium.com/@BrikerMan/multi-class-text-classification-with-kashgari-in-15mins-c3e744ce971d)

## Quick start

### Requirements and Installation
The project is based on Keras 2.2.0+ and Python 3.6+, because it is 2019 and type hints is cool.

```bash
pip install kashgari
# CPU
pip install tensorflow==1.12.0
# GPU
pip install tensorflow-gpu==1.12.0
```

### Example Usage
lets run a text classification with CNN model over [SMP 2017 ECDT Task1](http://ir.hit.edu.cn/smp2017ecdt-data).

```python
>>> from kashgari.corpus import SMP2017ECDTClassificationCorpus
>>> from kashgari.tasks.classification import CNNLSTMModel

>>> x_data, y_data = SMP2017ECDTClassificationCorpus.get_classification_data()
>>> x_data[0]
['你', '知', '道', '我', '几', '岁']
>>> y_data[0]
'chat'

# provided classification models `CNNModel`, `BLSTMModel`, `CNNLSTMModel` 
>>> classifier = CNNLSTMModel()
>>> classifier.fit(x_data, y_data)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 10)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 10, 100)           87500     
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 10, 32)            9632      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 5, 32)             0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               53200     
_________________________________________________________________
dense_1 (Dense)              (None, 32)                3232      
=================================================================
Total params: 153,564
Trainable params: 153,564
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5
 1/35 [..............................] - ETA: 32s - loss: 3.4652 - acc: 0.0469

... 

>>> x_test, y_test = SMP2017ECDTClassificationCorpus.get_classification_data('test')
>>> classifier.evaluate(x_test, y_test)
              precision    recall  f1-score   support
         
        calc       0.75      0.75      0.75         8
        chat       0.83      0.86      0.85       154
    contacts       0.54      0.70      0.61        10
    cookbook       0.97      0.94      0.95        89
    datetime       0.67      0.67      0.67         6
       email       1.00      0.88      0.93         8
         epg       0.61      0.56      0.58        36
      flight       1.00      0.90      0.95        21
...
```

### Run with GPT-2 Embedding

```python
from kashgari.embeddings import GPT2Embedding
from kashgari.tasks.classification import CNNLSTMModel
from kashgari.corpus import SMP2017ECDTClassificationCorpus

gpt2_embedding = GPT2Embedding('<path-to-gpt-model-folder>', sequence_length=30)                                 
model = CNNLSTMModel(gpt2_embedding)

train_x, train_y = SMP2017ECDTClassificationCorpus.get_classification_data()
model.fit(train_x, train_y)
```

### Run with Bert Embedding

```python
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import CNNLSTMModel
from kashgari.corpus import SMP2017ECDTClassificationCorpus

bert_embedding = BERTEmbedding('<bert-model-folder>', sequence_length=30)                                   
model = CNNLSTMModel(bert_embedding)

train_x, train_y = SMP2017ECDTClassificationCorpus.get_classification_data()
model.fit(train_x, train_y)
```

### Run with Word2vec Embedding

```python
from kashgari.embeddings import WordEmbeddings
from kashgari.tasks.classification import CNNLSTMModel
from kashgari.corpus import SMP2017ECDTClassificationCorpus

bert_embedding = WordEmbeddings('sgns.weibo.bigram', sequence_length=30)                                  
model = CNNLSTMModel(bert_embedding)
train_x, train_y = SMP2017ECDTClassificationCorpus.get_classification_data()
model.fit(train_x, train_y)
```

### Support for Training on Multiple GPUs

```python
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import CNNLSTMModel

train_x, train_y = prepare_your_classification_data()

# build model with embedding
bert_embedding = BERTEmbedding('bert-large-cased', sequence_length=128)
model = CNNLSTMModel(bert_embedding)

# or without pre-trained embedding
model = CNNLSTMModel()

# Build model with your corpus
model.build_model(train_x, train_y)

# Add multi gpu support
model.build_multi_gpu_model(gpus=8)

# Train, 256 / 8 = 32 samples for every GPU per batch
model.fit(train_x, train_y, batch_size=256)
```

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; start with the [contributor guidelines](CONTRIBUTING.md) and then check these open issues for specific tasks.

## Reference
This library is inspired by and references following frameworks and papers.

* [flair - A very simple framework for state-of-the-art Natural Language Processing (NLP)](https://github.com/zalandoresearch/flair)
* [anago - Bidirectional LSTM-CRF and ELMo for Named-Entity Recognition, Part-of-Speech Tagging](https://github.com/Hironsan/anago)
* [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FBrikerMan%2FKashgari.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FBrikerMan%2FKashgari?ref=badge_large)
