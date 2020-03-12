# Bare Embedding

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