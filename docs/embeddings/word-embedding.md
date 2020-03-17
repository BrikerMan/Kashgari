# Word Embedding

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