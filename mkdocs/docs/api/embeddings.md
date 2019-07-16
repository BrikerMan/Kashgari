# embeddings

## \_\_init\_\_

Embedding layers have its own \_\_init\_\_ function, check it out from their document page.

| class name                                                              | description                                                                 |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [BareEmbedding](../embeddings/bare-embedding.md)                        | random init `tf.keras.layers.Embedding` layer for text sequence embedding   |
| [WordEmbedding](../embeddings/word-embedding.md)                        | pre-trained Word2Vec embedding                                              |
| [BERTEmbedding](../embeddings/bert-embedding.md)                        | pre-trained BERT embedding                                                  |
| [GPT2Embedding](../embeddings/gpt2-embedding.md)                        | pre-trained GPT-2 embedding                                                 |
| [NumericFeaturesEmbedding](../embeddings/numeric-features-embedding.md) | random init `tf.keras.layers.Embedding` layer for numeric feature embedding |
| [StackedEmbedding](../embeddings/stacked-embeddingmd)                   | stack other embeddings for multi-input model                                |

All embedding layer shares same API except the `__init__` function.

## Properties

### token_count

int, corpus token count

### sequence_length

int, model sequence length

### label2idx

dict, label to index dict

### token_count

int, corpus token count

### tokenizer

Built-in Tokenizer of Embedding layer, available in `BERTEmbedding`.

## Methods

### analyze_corpus

Analyze data, build the token dict and label dict

```python
def analyze_corpus(self,
                   x: List[List[str]],
                   y: Union[List[List[str]], List[str]]):
```

__Args__:

- **x**: Array of input data
- **y_train**: Array of label data

### process\_x\_dataset

Batch process feature data to tensor, mostly call processor's `process_x_dataset` function to handle the data.

```python
def process_x_dataset(self,
                      data: List[List[str]],
                      subset: Optional[List[int]] = None) -> np.ndarray:
```

__Args__:

- **data**: target dataset
- **subset**: subset index list

__Returns__:

- vectorized feature tensor

### process\_y\_dataset

Batch process labels data to tensor, mostly call processor's `process_y_dataset` function to handle the data.

```python
def process_y_dataset(self,
                      data: List[List[str]],
                      subset: Optional[List[int]] = None) -> np.ndarray:
```

__Args__:

- **data**: target dataset
- **subset**: subset index list

__Returns__:

- vectorized label tensor

### reverse\_numerize\_label\_sequences

```python
def reverse_numerize_label_sequences(self,
                                     sequences,
                                     lengths=None):
```

### embed

Batch embed sentences, use this function for feature extraction. Input text then get the tensor representation.

```python
def embed(self,
          sentence_list: Union[List[List[str]], List[List[int]]],
          debug: bool = False) -> np.ndarray:
```

__Args__:

- **sentence_list**: Sentence list to embed
- **debug**: Show debug info, defualt False

__Returns__:

- A list of numpy arrays representing the embeddings

### embed_one

Dummy function for embed single sentence.

__Args__:

- **sentence**: Target sentence, list of tokens

__Returns__:

- Numpy arrays representing the embeddings

### info

Returns a dictionary containing the configuration of the model.

```python
def info(self) -> Dict:
```
