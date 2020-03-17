# Stacked Embedding

StackedEmbedding is a special kind of embedding class, which will able to stack other embedding layers togather for multi-input models.

More details checkout the example: [Handle Numeric features](../advance-use/handle-numeric-features.md)

```python
kashgari.embeddings.StackedEmbedding(embeddings: List[Embedding],
                                     processor: Optional[BaseProcessor] = None)
```

**Arguments**

- **embeddings**: list of embedding object.
