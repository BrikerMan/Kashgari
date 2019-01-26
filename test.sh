#!/usr/bin/env bash

export PYTHONPATH=`pwd`
python test/kashgari/test_corpus.py
python test/kashgari/embeddings/test_embeddings.py
python test/kashgari/tasks/classification/test_classification_models.py
python test/kashgari/tasks/seq_labeling/test_seq_labeling_models.py
