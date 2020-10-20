
.. _bert-embedding:

Bert Embedding
==============



BertEmbedding is a simple wrapped class of `Transformer Embedding <../transformer-embedding>`_. If you need load other kind of transformer based language model, please use the `Transformer Embedding <../transformer-embedding>`_.

.. note::
    When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding

.. autofunction:: kashgari.embeddings.BertEmbedding.__init__

Example Usage - Text Classification
-----------------------------------

Let's run a text classification model with BERT.

.. code-block:: python

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
   import os
   from kashgari.embeddings import BertEmbedding
   from kashgari.tokenizers import BertTokenizer

   bert_embedding = BertEmbedding('<PATH_TO_BERT_EMBEDDING>')

   tokenizer = BertTokenizer.load_from_vocab_file(os.path.join('<PATH_TO_BERT_EMBEDDING>', 'vocab_chinese.txt'))
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

   ########## build model ##########
   from kashgari.tasks.classification import CNN_LSTM_Model
   model = CNN_LSTM_Model(bert_embedding)

   ########## /build model ##########
   model.fit(
       train_x, train_y,
       validate_x, validate_y,
       epochs=3,
       batch_size=32
   )
   # save model
   model.save('path/to/save/model/to')

Use sentence pairs for input
----------------------------

let's assume input pair sample is ``"First do it" "then do it right"``\ , Then first tokenize the sentences using bert tokenizer. Then

.. code-block:: python

   sentence1 = ['First', 'do', 'it']
   sentence2 = ['then', 'do', 'it', 'right']

   sample = sentence1 + ["[SEP]"] + sentence2
   # Add a special separation token `[SEP]` between two sentences tokens
   # Generate a new token list
   # ['First', 'do', 'it', '[SEP]', 'then', 'do', 'it', 'right']

   train_x = [sample]
