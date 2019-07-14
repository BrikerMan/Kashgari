# Numeric Features Embedding

NumericFeaturesEmbedding is a random init `tf.keras.layers.Embedding` layer for numeric feature embedding. Which usally comes togather with [StackedEmbedding](./stacked-embedding.md) for represent non text features.

More details checkout the example: [Handle Numeric features](../advance-use/handle-numeric-features.md)

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