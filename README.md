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

## Road Map

- [x] Based on TensorFlow 2.0+ [@BrikerMan]
- [x] Fully support generator based training (#336 ,#273) [@BrikerMan]
- [ ] Clean code and full document
- [ ] Multi GPU/TPU Support [@BrikerMan]
- [x] Embeddings
    - [x] Bare Embedding [@BrikerMan]
    - [x] Word Embedding (Load trained W2V) [@BrikerMan]
    - [x] BERT Embedding (Based on bert4keras, support BERT, RoBERTa, ALBERT...) (#316) [@BrikerMan]
    - [x] GPT-2 Embedding
    - [ ] FeaturesEmbedding (Support Numeric feature as input)
    - [ ] Stacked Embedding (Stack Text embedding and features Embedding)
- [x] Classification Task
- [x] Labeling Task
- [x] Seq2Seq Task
- [ ] Built-in Callbacks
    - [x] Evaluate Callback
    - [ ] Save Best Callback
- [ ] Support TensorFlow Hub (Optional)
