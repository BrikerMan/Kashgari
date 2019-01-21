# Kashgari
Yet another state-of-the-art NLP framework with pre-trained embeddings and models.

Kashgare is:

# TODO
* **Powerful and simple NLP library**. Kashgare allows you to apply state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS), sense disambiguation and classification.

## Quick start
```bash
pip install kashgari
```

## Example Usage

lets run a text classification with CNN model over [Tencent Dingdang SLU Corpus](http://tcci.ccf.org.cn/conference/2018/taskdata.php).

### run classification

```python
>>> import kashgari as ks

>>> x_data, y_data = ks.corpus.TencentDingdangSLUCorpus.get_classification_data()
>>> x_data[0]
'导航结束'
>>> y_data[0]
'navigation.cancel_navigation'

# provided classification models `CNNModel`, `BLSTMModel`, `CNNLSTMModel` 
>>> classifier = ks.tasks.classification.CNNModel()
>>> classifier.fit(x_data, y_data)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 80)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 80, 100)           400300    
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 76, 128)           64128     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_2 (Dense)              (None, 11)                715       
=================================================================
Total params: 473,399
Trainable params: 473,399
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5

... 
```

## Run with pre-embedded word2vec

```python
import kashgari as ks

embedding = ks.embedding.Word2VecEmbedding(name_or_path='sgns.weibo.bigram')
tokenizer = ks.tokenizer.Tokenizer(embedding=embedding,
                                   sequence_length=30,
                                   segmenter=ks.k.SegmenterType.jieba)
                                   
model = ks.tasks.classification.CNNLSTMModel(tokenizer=tokenizer)
model.fit(x_data, y_data)
```

## Run with bert embedded word2vec

```python
import kashgari as ks

embedding = ks.embedding.BERTEmbedding('/Users/brikerman/Desktop/corpus/bert/chinese_L-12_H-768_A-12')
tokenizer = ks.tokenizer.Tokenizer(embedding=embedding,
                                   sequence_length=30,
                                   segmenter=ks.k.SegmenterType.jieba)
                                   
model = ks.tasks.classification.CNNLSTMModel(tokenizer=tokenizer)
model.fit(x_data, y_data)
```