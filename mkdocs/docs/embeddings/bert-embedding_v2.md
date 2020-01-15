# BERT Embedding V2

BERTEmbeddingV2 is based on [bert4keras](https://github.com/bojone/bert4keras). The embeddings itself are wrapped into our simple embedding interface so that they can be used like any other embedding.

BERTEmbeddingV2 support models:

| Model   | Author | Link                                                                       |     | Example |
| ------- | ------ | -------------------------------------------------------------------------- | --- | ------- |
| BERT    | Google | https://github.com/google-research/bert                                    |     |         |
| ALBERT  | Google | https://github.com/google-research/ALBERT                                  |     |         |
| ALBERT  | 徐亮   | https://github.com/brightmart/albert_zh                                    |     |         |
| RoBERTa | 徐亮   | https://github.com/brightmart/roberta_zh                                   |     |         |
| RoBERTa | 哈工大 | https://github.com/ymcui/Chinese-BERT-wwm                                  |     |         |
| RoBERTa | 苏建林 | https://github.com/ZhuiyiTechnology/pretrained-models                      |     |         |
| NEZHA   | Huawei | https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA |     |         |

!!! tip
When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding

```python
kashgari.embeddings.BERTEmbedding(vacab_path: str,
                                  config_path: str,
                                  checkpoint_path: str,
                                  bert_type: str = 'bert',
                                  task: str = None,
                                  sequence_length: Union[str, int] = 'auto',
                                  processor: Optional[BaseProcessor] = None,
                                  from_saved_model: bool = False):
```

**Arguments**

- **vacab_path**: path of model's `vacab.txt` file
- **config_path**: path of model's `model.json` file
- **checkpoint_path**: path of model's checkpoint file
- **task**: `kashgari.CLASSIFICATION` `kashgari.LABELING`. Downstream task type, If you only need to feature extraction, just set it as `kashgari.CLASSIFICATION`.
- **sequence_length**: `'auto'` or integer. When using `'auto'`, use the 95% of corpus length as sequence length. If using an integer, let's say `50`, the input output sequence length will set to 50.

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
# ------------ Load Bert Embedding ------------
import os
import kashgari
from kashgari.embeddings.bert_embedding_v2 import BERTEmbeddingV2
from kashgari.tokenizer import BertTokenizer

# Setup paths
model_folder = '/Users/brikerman/Desktop/nlp/language_models/albert_base'
checkpoint_path = os.path.join(model_folder, 'model.ckpt-best')
config_path = os.path.join(model_folder, 'albert_config.json')
vacab_path = os.path.join(model_folder, 'vocab_chinese.txt')

tokenizer = BertTokenizer.load_from_vacab_file(vacab_path)
embed = BERTEmbeddingV2(vacab_path, config_path, checkpoint_path,
                        bert_type='albert',
                        task=kashgari.CLASSIFICATION,
                        sequence_length=100)

sentences_tokenized = [tokenizer.tokenize(s) for s in sentences]
"""
The sentences will become tokenized into:
[
    ['jim', 'henson', 'was', 'a', 'puppet', '##eer', '.'],
    ['this', 'here', "'", 's', 'an', 'example', 'of', 'using', 'the', 'bert', 'token', '##izer', '.'],
    ['why', 'did', 'the', 'chicken', 'cross', 'the', 'road', '?']
]
"""

train_x, train_y = sentences_tokenized[:2], labels[:2]
validate_x, validate_y = sentences_tokenized[2:], labels[2:]

# ------------ Build Model Start ------------
from kashgari.tasks.classification import CNNLSTMModel
model = CNNLSTMModel(bert_embedding)

# ------------ Build Model End ------------

model.fit(
    train_x, train_y,
    validate_x, validate_y,
    epochs=3,
    batch_size=32
)
# save model
model.save('path/to/save/model/to')
```
