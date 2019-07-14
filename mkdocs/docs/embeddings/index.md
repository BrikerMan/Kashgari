# Language Embeddings

Kashgari provides several embeddings for language representation. Embedding layers will convert input sequence to tensor for downstream task. Availabel embeddings list:

| class name                                                  | description                                                                 |
| ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| [BareEmbedding](./bare-embedding.md)                        | random init `tf.keras.layers.Embedding` layer for text sequence embedding   |
| [WordEmbedding](./word-embedding.md)                        | pre-trained Word2Vec embedding                                              |
| [BERTEmbedding](./bert-embedding.md)                        | pre-trained BERT embedding                                                  |
| [GPT2Embedding](./gpt2-embedding.md)                        | pre-trained GPT-2 embedding                                                 |
| [NumericFeaturesEmbedding](./numeric-features-embedding.md) | random init `tf.keras.layers.Embedding` layer for numeric feature embedding |
| [StackedEmbedding](./stacked-embeddingmd)                   | stack other embeddings for multi-input model                                |

All embedding classes inherit from the `Embedding` class and implement the `embed()` to embed your input sequence and `embed_model` property which you need to build you own Model. By providing the `embed()` function and `embed_model` property, Kashgari hides the the complexity of different language embedding from users, all you need to care is which language embedding you need.

## Quick start

## Feature Extract From Pre-trained Embedding

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

## Classification and Labeling

See details at classification and labeling tutorial.

## Customized model

You can access the tf.keras model of embedding and add your own layers or any kind customizion. Just need to access the `embed_model` property of the embedding object.
