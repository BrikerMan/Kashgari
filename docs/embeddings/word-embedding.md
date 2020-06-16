# Word Embedding

```python
kashgari.embeddings.WordEmbedding(w2v_path: str,
                                  w2v_kwargs: Dict[str, Any] = None)
```

WordEmbedding is a `tf.keras.layers.Embedding` layer with pre-trained Word2Vec/GloVe Emedding weights.

**When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding**

**Arguments**

- **w2v_path**: Word2Vec file path.
- **w2v_kwargs**: params pass to the `load_word2vec_format()` function of `gensim.models.KeyedVectors` - https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
