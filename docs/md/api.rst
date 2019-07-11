.. _api:

API Documentation
=================

Documentations for the public classes and functions of ``kashgari``.

Labeling Models
---------------

.. py:currentmodule:: kashgari.tasks.labeling

.. autosummary::
    :nosignatures:

    BiLSTM_Model
    BiLSTM_CRF_Model

    BiGRU_Model
    BiGRU_CRF_Model

.. autoclass:: kashgari.tasks.labeling.BiLSTM_Model
   :members:
   :undoc-members:
   :inherited-members:

Embedding
---------

.. py:currentmodule:: kashgari.embeddings

.. autosummary::
    :nosignatures:

    BareEmbedding
    BERTEmbedding
    WordEmbedding
    NumericFeaturesEmbedding
    StackedEmbedding

.. automodule:: kashgari.embeddings
    :members:
    :undoc-members:
    :inherited-members:

Corpus
------

.. automodule:: kashgari.corpus
    :members:
    :undoc-members:
    :inherited-members: