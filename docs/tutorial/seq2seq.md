# Seq2Seq

## Train a translate model

```python
# Original Corpus
x_original = [
    'Who am I?',
    'I am sick.',
    'I like you.',
    'I need help.',
    'It may hurt.',
    'Good morning.']

y_original = [
    'مەن كىم ؟',
    'مەن كېسەل.',
    'مەن سىزنى ياخشى كۆرمەن',
    'ماڭا ياردەم كېرەك.',
    'ئاغىرىشى مۇمكىن.',
    'خەيىرلىك ئەتىگەن.']

# Tokenize sentence with custom tokenizing function
# Tokenize sentence with custom tokenizing function
# We use Bert Tokenizer for this demo
from kashgari.tokenizers import BertTokenizer
tokenizer = BertTokenizer()
x_tokenized = [tokenizer.tokenize(sample) for sample in x_original]
y_tokenized = [tokenizer.tokenize(sample) for sample in y_original]
```

After tokenizing the corpus, we can build a seq2seq Model.

```python
from kashgari.tasks.seq2seq import Seq2Seq

model = Seq2Seq()
model.fit(x_tokenized, y_tokenized)

# predict with model
preds, attention = model.predict(x_tokenized)
print(preds)
```

## Train with custom embedding

You can define both encoder's and decoder's embedding. This is how to use [Bert Embedding](./../embeddings/bert-embedding.md) as encoder's embedding layer.

```python
from kashgari.embeddings import BertEmbedding
bert = BertEmbedding('<Path-to-bert-embedding>')

model = Seq2Seq(encoder_embedding=bert, hidden_size=512)
model.fit(x_tokenized, y_tokenized)
```