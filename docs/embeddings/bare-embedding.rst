
Bare Embedding
==============

BareEmbedding is a random init ``tf.keras.layers.Embedding`` layer for text sequence embedding, which is the defualt embedding class for kashgari models.

.. autofunction:: kashgari.embeddings.BareEmbedding.__init__

Here is the sample how to use embedding class. The key difference here is that must call ``analyze_corpus`` function before using the embed function. This is because the embedding layer is not pre-trained and do not contain any word-list. We need to build word-list from the corpus.

.. code-block:: python

   import kashgari
   from kashgari.embeddings import BareEmbedding

   embedding = BareEmbedding(embedding_size=100)

   embedding.analyze_corpus(x_data, y_data)

   embed_tensor = embedding.embed_one(['语', '言', '模', '型'])
