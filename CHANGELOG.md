## v0.2.6 (2019.07.12)

* Fixing lstm issue in labeling model (#125)
* Add tf.keras version info.

## v0.2.4 (2019.06.06)

* Add BERT output feature layer fine-tune support. Discussion: #103
* Add BERT output feature layer number selection, default 4 according to BERT paper.
* Fix BERT embedding token index offset issue #104.

## v0.2.1 (2019.03.05)

* fix missing `sequence_labeling_tokenize_add_bos_eos` consig

## v0.2.0

* multi-label classification for all classification models
* support cuDNN cell for sequence labeling
* add option for output `BOS` and `EOS` in sequence labeling result, fix #31 

## v0.1.9

* add `AVCNNModel`, `KMaxCNNModel`, `RCNNModel`, `AVRNNModel`, `DropoutBGRUModel`, `DropoutAVRNNModel` model to classification task.
* fix several small bugs

## v0.1.8
* fix BERT Embedding  model's `to_json` function, issue #19 

## v0.1.7

* remove class candidates filter to fix #16
* overwrite init function in CustomEmbedding
* add parameter check to custom_embedding layer
* add `keras-bert` version to setup.py file

## v0.1.6
* add `output_dict`, `debug_info` params to text_classification model
* add `output_dict`, `debug_info` and `chunk_joiner `params to text_classification model
* fix possible crash at data_generator

## v0.1.5

* fix sequence labeling evaluate result output
* refactor model save and load function

## v0.1.4

* fix classification model evaluate result output
* change test settings