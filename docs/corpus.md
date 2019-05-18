# Corpus

Kashgari provides several build-in corpus for testing.

## Chinese Daily Ner Corpus
Chinese Ner corpus cotains 20864 train samples, 4636 test samples and 2318 valid samples.

Usage:

```python
from kashgari.corpus import ChineseDailyNerCorpus

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
```

Data Sample:

```python
>>> x[0] 
['海', '钓', '比', '赛', '地', '点', '在', '厦', '门']
>>> y[0] 
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC']
```

## SMP2018 ECDT Human-Computer Dialogue Classification Corpus

https://worksheets.codalab.org/worksheets/0x27203f932f8341b79841d50ce0fd684f/

This dataset is released by the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2018-ECDT)
task 1 and is provided by the iFLYTEK Corporation, which is a Chinese human-computer dialogue dataset.

```
      label           query
0   weather        今天东莞天气如何
1       map  从观音桥到重庆市图书馆怎么走
2  cookbook          鸭蛋怎么腌？
3    health         怎么治疗牛皮癣
4      chat             唠什么
```

Usage:

```python
from kashgari.corpus import SMP2018ECDTCorpus

train_x, train_y = SMP2018ECDTCorpus.load_data('train')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')

# Change cutter to jieba, need to install jieba first
train_x, train_y = SMP2018ECDTCorpus.load_data('train', cutter='jieba')
test_x, test_y = SMP2018ECDTCorpus.load_data('test', cutter='jieba')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid', cutter='jieba')
```

Data Sample:

```python
# char cutted
>>> x[0] 
[['给', '周', '玉', '发', '短', '信']]
>>> y[0] 
['message']

# jieba cutted
>>> x[0] 
[['给', '周玉', '发短信']]
>>> y[0] 
['message']
```