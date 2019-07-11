# Language Embeddings

Kashgari provides several embeddings for language representation. Embedding layers will convert input sequence to tensor for downstream task. Availabel embeddings list:

| class name                                              | description                                                                 |
| ------------------------------------------------------- | --------------------------------------------------------------------------- |
| [BareEmbedding](#bare-embedding)                        | random init `tf.keras.layers.Embedding` layer for text sequence embedding   |
| [WordEmbedding](#word-embedding)                        | pre-trained Word2Vec embedding                                              |
| [BERTEmbedding](#bert-embedding)                        | pre-trained BERT embedding                                                  |
| [GPT2Embedding](#gpt2-embedding)                        | pre-trained GPT-2 embedding                                                 |
| [NumericFeaturesEmbedding](#numeric-features-embedding) | random init `tf.keras.layers.Embedding` layer for numeric feature embedding |
| [StackedEmbedding](#stacked-embedding)                  | stack other embeddings for multi-input model                                |

All embedding classes inherit from the `Embedding` class and implement the `embed()` to embed your input sequence and `embed_model` property which you need to build you own Model. By providing the `embed()` function and `embed_model` property, Kashgari hides the the complexity of different language embedding from users, all you need to care is which language embedding you need.

## Quick start

### Feature Extract From Pre-trained Embedding

Feature Extraction is one of the major way to use pre-trained language embedding. Kashgari provides simple API for this task. All you need to is init a embedding object then call `embed` function. Here is the example. All embedding shares same embed API.

```python
import kashgari
from kashgari.embeddings import BERTEmbedding

# need to spesify task for the downstream task,
# if use embedding for feature extraction, just set `task=kashgari.CLASSIFICATION`
bert = BERTEmbedding('<BERT_MODEL_FOLDER>',
                     task=kashgari.CLASSIFICATION,
                     sequence_length=100)
# call for bulk embed
embed_tensor = bert.embed([['语', '言', '模', '型']])

# call for single embed
embed_tensor = bert.embed_one(['语', '言', '模', '型'])

print(embed_tensor)
# array([[-0.5001117 ,  0.9344998 , -0.55165815, ...,  0.49122602,
#         -0.2049343 ,  0.25752577],
#        [-1.05762   , -0.43353617,  0.54398274, ..., -0.61096823,
#          0.04312163,  0.03881482],
#        [ 0.14332692, -0.42566583,  0.68867105, ...,  0.42449307,
#          0.41105768,  0.08222893],
#        ...,
#        [-0.86124015,  0.08591427, -0.34404194, ...,  0.19915134,
#         -0.34176797,  0.06111742],
#        [-0.73940575, -0.02692179, -0.5826528 , ...,  0.26934686,
#         -0.29708537,  0.01855129],
#        [-0.85489404,  0.007399  , -0.26482674, ...,  0.16851354,
#         -0.36805922, -0.0052386 ]], dtype=float32)
```

### Classification and Labeling

See details at classification and labeling tutorial.

### Customized model

You can access the tf.keras model of embedding and add your own layers or any kind customizion. Just need to access the `embed_model` property of the embedding object.

## Bare Embedding

```python
kashgari.embeddings.BareEmbedding(task: str = None,
                                  sequence_length: Union[int, str] = 'auto',
                                  embedding_size: int = 100,
                                  processor: Optional[BaseProcessor] = None)
```

BareEmbedding is a random init `tf.keras.layers.Embedding` layer for text sequence embedding, which is the defualt embedding class for kashgari models.

**Arguments**

- **task**: `kashgari.CLASSIFICATION` `kashgari.LABELING`. Downstream task type, If you only need to feature extraction, just set it as `kashgari.CLASSIFICATION`.
- **sequence_length**: `'auto'`, `'variable'` or integer. When using `'auto'`, use the 95% of corpus length as sequence length. When using `'variable'`, model input shape will set to None, which can handle various length of input, it will use the length of max sequence in every batch for sequence length. If using an integer, let's say `50`, the input output sequence length will set to 50.
- **embedding_size**: Dimension of the dense embedding.

Here is the sample how to use embedding class. The key difference here is that must call `analyze_corpus` function before using the embed function. This is because the embedding layer is not pre-trained and do not contain any word-list. We need to build word-list from the corpus.

```python
import kashgari
from kashgari.embeddings import BareEmbedding

embedding = BareEmbedding(task=kashgari.CLASSIFICATION,
                          sequence_length=100,
                          embedding_size=100)

embedding.analyze_corpus(x_data, y_data)

embed_tensor = embedding.embed_one(['语', '言', '模', '型'])
```

## Word Embedding

```python
kashgari.embeddings.WordEmbedding(w2v_path: str,
                                  task: str = None,
                                  w2v_kwargs: Dict[str, Any] = None,
                                  sequence_length: Union[Tuple[int, ...], str, int] = 'auto',
                                  processor: Optional[BaseProcessor] = None)
```

WordEmbedding is a `tf.keras.layers.Embedding` layer with pre-trained Word2Vec/GloVe Emedding weights.

**When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding**

**Arguments**

- **w2v_path**: Word2Vec file path.
- **task**: `kashgari.CLASSIFICATION` `kashgari.LABELING`. Downstream task type, If you only need to feature extraction, just set it as `kashgari.CLASSIFICATION`.
- **w2v_kwargs**: params pass to the `load_word2vec_format()` function of `gensim.models.KeyedVectors` - https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
- **sequence_length**: `'auto'`, `'variable'` or integer. When using `'auto'`, use the 95% of corpus length as sequence length. When using `'variable'`, model input shape will set to None, which can handle various length of input, it will use the length of max sequence in every batch for sequence length. If using an integer, let's say `50`, the input output sequence length will set to 50.

## BERT Embedding

BERTEmbedding is based on [keras-bert](https://github.com/CyberZHG/keras-bert). The embeddings itself are wrapped into our simple embedding interface so that they can be used like any other embedding.

BERTEmbedding support BERT variants like **ERNIE**, but need to load the **tensorflow checkpoint**. If you intrested to use ERNIE, just download [tensorflow_ernie](https://github.com/ArthurRizar/tensorflow_ernie) and load like BERT Embedding.

**When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding**

```python

kashgari.embeddings.BERTEmbedding(model_folder: str,
                                  layer_nums: int = 4,
                                  trainable: bool = False,
                                  task: str = None,
                                  sequence_length: Union[str, int] = 'auto',
                                  processor: Optional[BaseProcessor] = None)
```

**Arguments**

- **model_folder**: path of checkpoint folder.
- **layer_nums**: number of layers whose outputs will be concatenated into a single tensor, default `4`, output the last 4 hidden layers as the thesis suggested.
- **trainable**: whether if the model is trainable, default `False` and set it to `True` for fine-tune this embedding layer during your training.
- **task**: `kashgari.CLASSIFICATION` `kashgari.LABELING`. Downstream task type, If you only need to feature extraction, just set it as `kashgari.CLASSIFICATION`.
- **sequence_length**: `'auto'`, `'variable'` or integer. When using `'auto'`, use the 95% of corpus length as sequence length. When using `'variable'`, model input shape will set to None, which can handle various length of input, it will use the length of max sequence in every batch for sequence length. If using an integer, let's say `50`, the input output sequence length will set to 50.

## GPT2 Embedding

GPT2Embedding is based on [keras-gpt-2](https://github.com/CyberZHG/keras-gpt-2). The embeddings itself are wrapped into our simple embedding interface so that they can be used like any other embedding.

**When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding**

```python

kashgari.embeddings.GPT2Embedding(model_folder: str,
                                  task: str = None,
                                  sequence_length: Union[str, int] = 'auto',
                                  processor: Optional[BaseProcessor] = None)
```

**Arguments**

- **model_folder**: path of checkpoint folder.
- **task**: `kashgari.CLASSIFICATION` `kashgari.LABELING`. Downstream task type, If you only need to feature extraction, just set it as `kashgari.CLASSIFICATION`.
- **sequence_length**: `'auto'`, `'variable'` or integer. When using `'auto'`, use the 95% of corpus length as sequence length. When using `'variable'`, model input shape will set to None, which can handle various length of input, it will use the length of max sequence in every batch for sequence length. If using an integer, let's say `50`, the input output sequence length will set to 50.

## Numeric Features Embedding

NumericFeaturesEmbedding is a random init `tf.keras.layers.Embedding` layer for numeric feature embedding. Which usally comes togather with [StackedEmbedding](#stacked-embedding) for represent non text features.

More details checkout the example: [Handle Numeric features](./deal_with_numeric_features.html)

```python
kashgari.embeddings.NumericFeaturesEmbedding(feature_count: int,
                                             feature_name: str,
                                             sequence_length: Union[str, int] = 'auto',
                                             embedding_size: int = None,
                                             processor: Optional[BaseProcessor] = None)
```

**Arguments**

- **feature_count**: count of the features of this embedding.
- **feature_name**: name of the feature.
- **sequence_length**: `'auto'`, `'variable'` or integer. When using `'auto'`, use the 95% of corpus length as sequence length. When using `'variable'`, model input shape will set to None, which can handle various length of input, it will use the length of max sequence in every batch for sequence length. If using an integer, let's say `50`, the input output sequence length will set to 50.
- **embedding_size**: Dimension of the dense embedding.

## Stacked Embedding

StackedEmbedding is a special kind of embedding class, which will able to stack other embedding layers togather for multi-input models.

More details checkout the example: [Handle Numeric features](./deal_with_numeric_features.html)

```python
kashgari.embeddings.StackedEmbedding(embeddings: List[Embedding],
                                     processor: Optional[BaseProcessor] = None)
```

**Arguments**

- **embeddings**: list of embedding object.
