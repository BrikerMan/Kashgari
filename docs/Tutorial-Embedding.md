# Tutorial 3: Word Embeddings

Kashgari provide classic Word2vec embedding and BERT embedding to embed the words in sentences.


## Embeddings

All word embedding classes inherit from the `BaseEmbedding` class and implement the `embed()` to embed your text and `model` property which you need to fine-tune a keras model. This mean you for most users of Kashgari don't, the complexity of different embeddings remains hidden behind this interface. 

All embeddings produced with kashgari embed method is a numpy array with fixed shape, so they can be immediately used for training and fine-tuning. Or you could use the `embedding.model` property which provides a functional keras model, and add your layer to this model.

## Classic Word Embedding

Classic word embeddings are static and word-level, meaning that each distinct word gets exactly one pre-computed embedding. Most embeddings fall under this class, including the popular GloVe or Komnios embeddings. 

Simply instantiate the WordEmbeddings class and pass a string identifier or file path of the embedding you wish to load.

```python
from kashgari.embeddings import WordEmbeddings

# init embedding
embedding = WordEmbeddings('sgns.renmin.bigram', sequence_length=30)

# if the word2vec embedding is too big, you cloud add a limit arg to limit the size of vectors
# this will give you only the first 1000 vector

embedding_1000 = WordEmbeddings('sgns.renmin.bigram', sequence_length=30, limit=1000)
```

Now, create an example sentence and call the embedding's `embed()` method. You can also pass a list of sentences to this method since some embedding types make use of batching to increase speed.

```python
# create sentence.
>>> sentence = ['Muhammed', 'al-Kashgari', 'was', 'an', '11th-century', 'Kara-Khanid', 'scholar', 'and', 'lexicographer', 'from', 'Kashgar.']

>>> embedded_vector = embedding.embed(sentence)
>>> print(embedded_vector.shape)
(30, 300)
>>> print(embedded_vector)
array([[ 0.3689207 ,  0.13478056,  0.26002103, ...,  0.0961755 ,
         0.44058734,  0.25040022],
       [ 0.3689207 ,  0.13478056,  0.26002103, ...,  0.0961755 ,
         0.44058734,  0.25040022],
       [ 0.515267  ,  0.076714  , -0.116043  , ..., -0.040003  ,
         0.48516   ,  0.53112   ],
       ...,
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ]], dtype=float32)
```

Here is the list of embedding id, Thanks [Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)

| ID  | Language | Embedding |
| --- | -------- | --------- |
| sgns.renmin.bigram | Chinese | People's Daily News Word + Ngram |
| sgns.renmin.bigram-char | Chinese | People's Daily News Word + Character + Ngram |
| sgns.weibo.bigram | Chinese | Sina Weibo Word + Ngram |
| sgns.weibo.bigram-char | Chinese | Sina Weibo Word + Character + Ngram |

Now, this is the example of how to use the `model` property to fine-tune a model.

```python
>>> from keras.layers import Dense
>>> from keras.models import Model
>>> base_model = embedding.model
>>> base_model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 30)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 30, 300)           58561800  
=================================================================
Total params: 58,561,800
Trainable params: 0
Non-trainable params: 58,561,800
_________________________________________________________________

>>> dense_layer = Dense(12)(base_model.output)
>>> model = Model(base_model.input, dense_layer)
>>> model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 30)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 30, 300)           58561800  
_________________________________________________________________
dense_1 (Dense)              (None, 30, 12)            3612      
=================================================================
Total params: 58,565,412
Trainable params: 3,612
Non-trainable params: 58,561,800
_________________________________________________________________
```


## BERT embedding

[BERT embeddings](https://arxiv.org/pdf/1810.04805.pdf) were developed by Devlin et al. (2018) and are a different kind of powerful word embedding based on a bidirectional transformer architecture.
Kashgari is using the implementation of [CyberZHG](https://github.com/CyberZHG/keras-bert).
The embeddings itself are wrapped into our simple embedding interface, so that they can be used like any other
embedding.

```python
>>> from kashgari.embeddings import BERTEmbedding

# init embedding
>>> embedding = BERTEmbedding('bert-base-chinese', sequence_length=30)
```

You can load any of the pre-trained BERT models by providing the model string during initialization:

| ID | Language | Embedding |
| -------------     | ------------- | ------------- |
| 'bert-base-uncased' | English | 12-layer, 768-hidden, 12-heads, 110M parameters |
| 'bert-large-uncased'   | English | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| 'bert-base-cased'    | English | 12-layer, 768-hidden, 12-heads , 110M parameters |
| 'bert-large-cased'   | English | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| 'bert-base-multilingual-cased'     | 104 languages | 12-layer, 768-hidden, 12-heads, 110M parameters |
| 'bert-base-chinese'    | Chinese Simplified and Traditional | 12-layer, 768-hidden, 12-heads, 110M parameters |

## Custom Embedding
If you don't need a pre-trained embedding layer, you could use the `CustomEmbedding` class.

```python
>>> from kashgari.embeddings import CustomEmbedding

# init embedding
>>> embedding = CustomEmbedding('custom_embedding', sequence_length=30, embedding_size=100)
>>> corpus = [['Muhammed', 'al-Kashgari', 'was', 'an', '11th-century', 'Kara-Khanid', 'scholar', 'and', 'lexicographer', 'from', 'Kashgar.']]
# build token2idx dict by corpus
>>> embedding.build_token2idx_dict(corpus, 3)
```