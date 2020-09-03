
.. transformer-embedding:

Transformer Embedding
=====================

TransformerEmbedding is based on `bert4keras <https://github.com/bojone/bert4keras>`_. The embeddings itself are wrapped into our simple embedding interface so that they can be used like any other embedding.

TransformerEmbedding support models:

.. list-table::
   :header-rows: 1

   * - Model
     - Author
     - Link
   * - BERT
     - Google
     - https://github.com/google-research/bert
   * - ALBERT
     - Google
     - https://github.com/google-research/ALBERT
   * - ALBERT
     - brightmart
     - https://github.com/brightmart/albert_zh
   * - RoBERTa
     - brightmart
     - https://github.com/brightmart/roberta_zh
   * - RoBERTa
     - 哈工大
     - https://github.com/ymcui/Chinese-BERT-wwm
   * - RoBERTa
     - 苏剑林
     - https://github.com/ZhuiyiTechnology/pretrained-models
   * - NEZHA
     - Huawei
     - https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA

.. note::
    When using pre-trained embedding, remember to use same tokenize tool with the embedding model, this will allow to access the full power of the embedding

.. autofunction:: kashgari.embeddings.TransformerEmbedding.__init__

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
   # ------------ Load Bert Embedding ------------
   import os
   from kashgari.embeddings import TransformerEmbedding
   from kashgari.tokenizers import BertTokenizer

   # Setup paths
   model_folder = '/xxx/xxx/albert_base'
   checkpoint_path = os.path.join(model_folder, 'model.ckpt-best')
   config_path = os.path.join(model_folder, 'albert_config.json')
   vocab_path = os.path.join(model_folder, 'vocab_chinese.txt')

   tokenizer = BertTokenizer.load_from_vocab_file(vocab_path)
   embed = TransformerEmbedding(vocab_path, config_path, checkpoint_path,
                                bert_type='albert')

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
   from kashgari.tasks.classification import CNN_LSTM_Model
   model = CNN_LSTM_Model(embed)

   # ------------ Build Model End ------------

   model.fit(
       train_x, train_y,
       validate_x, validate_y,
       epochs=3,
       batch_size=32
   )
   # save model
   model.save('path/to/save/model/to')
