# Embeddings

Kashgari provides several embeddings for language representation. Embedding layers will convert input sequence to tensor for downstream task. Availabel embeddings list:

| class name               | description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| BareEmbedding            | random init `tf.keras.layers.Embedding` layer for text sequence embedding   |
| WordEmbedding            | pre-trained Word2Vec embedding                                              |
| BERTEmbedding            | pre-trained BERT embedding                                                  |
| GPT2Embedding            | pre-trained GPT-2 embedding                                                 |
| NumericFeaturesEmbedding | random init `tf.keras.layers.Embedding` layer for numeric feature embedding |
| StackedEmbedding         | stack other embeddings for multi-input model                                |

All embedding classes inherit from the `Embedding` class and implement the `embed()` to embed your input sequence and `embed_model` property which you need to build you own Model. By providing the `embed()` function and `embed_model` property, Kashgari hides the the complexity of different language embedding from users, all you need to care is which language embedding you need.

## Bare Embedding

BareEmbedding is a random init `tf.keras.layers.Embedding` layer for text sequence embedding, which is the defualt embedding class for kashgari models.

Here is the sample how to use embedding class.

```python


```