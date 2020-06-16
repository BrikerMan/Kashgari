# BERT Embedding

## TODO: update to the latest API

BERTEmbedding is based on [keras-bert](https://github.com/CyberZHG/keras-bert). The embeddings itself are wrapped into our simple embedding interface so that they can be used like any other embedding.

BERTEmbedding support BERT variants like **ERNIE**, but need to load the **tensorflow checkpoint**. If you intrested to use ERNIE, just download [tensorflow_ernie](https://github.com/ArthurRizar/tensorflow_ernie) and load like BERT Embedding.

!!! tip
    When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding

```python
kashgari.embeddings.BertEmbedding(model_folder: str,
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

## Example Usage - Text Classification

Let's run a text classification model with BERT.

```python
sentences = [
    "Jim Henson was a puppeteer.",
    "This here's an example of using the BERT tokenizer.",
    "Why did the chicken cross the road?"
            ]
labels = [
    "class1",
    "class2",
    "class1"
]
########## Load Bert Embedding ##########
import kashgari
from kashgari.embeddings import BERTEmbedding

bert_embedding = BERTEmbedding(bert_model_path)

tokenizer = bert_embedding.tokenizer
sentences_tokenized = []
for sentence in sentences:
    sentence_tokenized = tokenizer.tokenize(sentence)
    sentences_tokenized.append(sentence_tokenized)
"""
The sentences will become tokenized into:
[
    ['[CLS]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '.', '[SEP]'],
    ['[CLS]', 'this', 'here', "'", 's', 'an', 'example', 'of', 'using', 'the', 'bert', 'token', '##izer', '.', '[SEP]'],
    ['[CLS]', 'why', 'did', 'the', 'chicken', 'cross', 'the', 'road', '?', '[SEP]']
]
"""

# Our tokenizer already added the BOS([CLS]) and EOS([SEP]) token
# so we need to disable the default add_bos_eos setting.
bert_embedding.processor.add_bos_eos = False

train_x, train_y = sentences_tokenized[:2], labels[:2]
validate_x, validate_y = sentences_tokenized[2:], labels[2:]

########## build model ##########
from kashgari.tasks.classification import CNNLSTMModel
model = CNNLSTMModel(bert_embedding)

########## /build model ##########
model.fit(
    train_x, train_y,
    validate_x, validate_y,
    epochs=3,
    batch_size=32
)
# save model
model.save('path/to/save/model/to')
```

## Use sentence pairs for input

let's assume input pair sample is `"First do it" "then do it right"`, Then first tokenize the sentences using bert tokenizer. Then

```python
sentence1 = ['First', 'do', 'it']
sentence2 = ['then', 'do', 'it', 'right']

sample = sentence1 + ["[SEP]"] + sentence2
# Add a special separation token `[SEP]` between two sentences tokens
# Generate a new token list
# ['First', 'do', 'it', '[SEP]', 'then', 'do', 'it', 'right']

train_x = [sample]
```

## Pre-trained models

| model            | provider             | Language       | Link             | info                          |
| ---------------- | -------------------- | -------------- | ---------------- | ----------------------------- |
| BERT official    | Google               | Multi Language | [link][bert]     |                               |
| ERNIE            | Baidu                | Chinese        | [link][ernie]    | Unofficial Tensorflow Version |
| Chinese BERT WWM | 哈工大讯飞联合实验室 | Chinese        | [link][bert-wwm] | Use Tensorflow Version        |

[bert]: https://github.com/google-research/bert
[ernie]: https://github.com/ArthurRizar/tensorflow_ernie
[bert-wwm]: https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD
