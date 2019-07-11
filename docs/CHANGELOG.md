# Changelog

## [0.5.0] - 2019.07.11

🎉🎉 tf.keras version 🎉🎉

- 🎉 Rewrite Kashgari using `tf.keras`. Discussion: [#77](https://github.com/BrikerMan/Kashgari/issues/77)
- 🎉 Rewrite Documents.
- ✨ Add TPU support.
- ✨ Add TF-Serving support.
- ✨ Add advance customization support, like multi-input model.
- 🐎 Performance optimization.

## [0.2.4] - 2019.06.06

- Add BERT output feature layer fine-tune support. Discussion: [#103](https://github.com/BrikerMan/Kashgari/issues/103)
- Add BERT output feature layer number selection, default 4 according to BERT paper.
- Fix BERT embedding token index offset issue [#104](https://github.com/BrikerMan/Kashgari/issues/104).

## [0.2.1] - 2019.03.05

- fix missing `sequence_labeling_tokenize_add_bos_eos` consig

## [0.2.0]

- multi-label classification for all classification models
- support cuDNN cell for sequence labeling
- add option for output `BOS` and `EOS` in sequence labeling result, fix #31 

## 0.1.9

- add `AVCNNModel`, `KMaxCNNModel`, `RCNNModel`, `AVRNNModel`, `DropoutBGRUModel`, `DropoutAVRNNModel` model to classification task.
- fix several small bugs

## 0.1.8

- fix BERT Embedding  model's `to_json` function, issue #19 

## 0.1.7

- remove class candidates filter to fix #16
- overwrite init function in CustomEmbedding
- add parameter check to custom_embedding layer
- add `keras-bert` version to setup.py file

## 0.1.6

- add `output_dict`, `debug_info` params to text_classification model
- add `output_dict`, `debug_info` and `chunk_joiner `params to text_classification model
- fix possible crash at data_generator

## 0.1.5

- fix sequence labeling evaluate result output
- refactor model save and load function

## 0.1.4

- fix classification model evaluate result output
- change test settings

[0.5.0]: https://github.com/BrikerMan/Kashgari/compare/8c1580c6ae44ce762b99474220cfdb72c4b68b45...v0.5.0
[0.2.4]: https://github.com/BrikerMan/Kashgari/compare/v0.2.1...v0.2.4
[0.2.1]: https://github.com/BrikerMan/Kashgari/compare/v0.2.0...v0.2.1