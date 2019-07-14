# GPT2 Embedding

GPT2Embedding is based on [keras-gpt-2](https://github.com/CyberZHG/keras-gpt-2). The embeddings itself are wrapped into our simple embedding interface so that they can be used like any other embedding.

!!! tip
    When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding

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
