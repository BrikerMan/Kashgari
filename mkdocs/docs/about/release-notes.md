# Release notes

## Upgrading

To upgrade Material to the latest version, use `pip`:

``` sh
pip install --upgrade kashgari-tf
```

To inspect the currently installed version, use the following command:

``` sh
pip show kashgari-tf
```

## Current Release

### [0.5.4] - 2019.09.30

- ✨ Add shuffle parameter to fit function ([#249])
- ✨ Improved type hinting for loaded model ([#248])
- 🐛 Fix loading models with CRF layers ([#244], [#228])
- 🐛 Fix the configuration changes during embedding save/load ([#224])
- 🐛 Fix stacked embedding save/load ([#224])
- 🐛 Fix evaluate function where the list has int instead of str ([#222])
- 💥 Renaming model.pre_processor to model.processor
- 🚨 Removing TensorFlow and numpy warnings
- 📝 Add docs how to specify which CPU or GPU
- 📝 Add docs how to compile model with custom optimizer

### [0.5.3] - 2019.08.11

- 🐛 Fixing CuDNN Error ([#198])

### [0.5.2] - 2019.08.10

- 💥 Add CuDNN Cell config, disable auto CuDNN cell. ([#182], [#198])

### [0.5.1] - 2019.07.15

- 📝 Rewrite documents with mkdocs
- 📝 Add Chinese documents
- ✨ Add `predict_top_k_class` for classification model to get predict probabilities ([#146](https://github.com/BrikerMan/Kashgari/issues/146))
- 🚸 Add `label2idx`, `token2idx` properties to Embeddings and Models
- 🚸 Add `tokenizer` property for BERT Embedding. ([#136](https://github.com/BrikerMan/Kashgari/issues/136))
- 🚸 Add `predict_kwargs` for models `predict()` function
- ⚡️ Change multi-label classification's default loss function to binary_crossentropy ([#151](https://github.com/BrikerMan/Kashgari/issues/151))

### [0.5.0] - 2019.07.11

🎉🎉 tf.keras version 🎉🎉

- 🎉 Rewrite Kashgari using `tf.keras` ([#77](https://github.com/BrikerMan/Kashgari/issues/77))
- 🎉 Rewrite Documents
- ✨ Add TPU support
- ✨ Add TF-Serving support.
- ✨ Add advance customization support, like multi-input model
- 🐎 Performance optimization

## Legacy Version Changelog

### [0.2.6] - 2019.07.12

- 📝 Add tf.keras version info
- 🐛 Fixing lstm issue in labeling model ([#125](https://github.com/BrikerMan/Kashgari/issues/125))

### [0.2.4] - 2019.06.06

- Add BERT output feature layer fine-tune support. Discussion: ([#103](https://github.com/BrikerMan/Kashgari/issues/103))
- Add BERT output feature layer number selection, default 4 according to BERT paper
- Fix BERT embedding token index offset issue ([#104](https://github.com/BrikerMan/Kashgari/issues/104)

### [0.2.1] - 2019.03.05

- fix missing `sequence_labeling_tokenize_add_bos_eos` config

### 0.2.0

- multi-label classification for all classification models
- support cuDNN cell for sequence labeling
- add option for output `BOS` and `EOS` in sequence labeling result, fix #31 

### 0.1.9

- add `AVCNNModel`, `KMaxCNNModel`, `RCNNModel`, `AVRNNModel`, `DropoutBGRUModel`, `DropoutAVRNNModel` model to classification task.
- fix several small bugs

### 0.1.8

- fix BERT Embedding  model's `to_json` function, issue #19 

### 0.1.7

- remove class candidates filter to fix #16
- overwrite init function in CustomEmbedding
- add parameter check to custom_embedding layer
- add `keras-bert` version to setup.py file

### 0.1.6

- add `output_dict`, `debug_info` params to text_classification model
- add `output_dict`, `debug_info` and `chunk_joiner `params to text_classification model
- fix possible crash at data_generator

### 0.1.5

- fix sequence labeling evaluate result output
- refactor model save and load function

### 0.1.4

- fix classification model evaluate result output
- change test settings

[0.5.4]: https://github.com/BrikerMan/Kashgari/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/BrikerMan/Kashgari/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/BrikerMan/Kashgari/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/BrikerMan/Kashgari/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/BrikerMan/Kashgari/compare/milestone/tf.keras...v0.5.0
[0.2.6]: https://github.com/BrikerMan/Kashgari/compare/v0.2.4...v0.2.6
[0.2.4]: https://github.com/BrikerMan/Kashgari/compare/v0.2.1...v0.2.4
[0.2.1]: https://github.com/BrikerMan/Kashgari/compare/v0.2.0...v0.2.1

[#182]: https://github.com/BrikerMan/Kashgari/issues/182
[#198]: https://github.com/BrikerMan/Kashgari/issues/198
[#224]: https://github.com/BrikerMan/Kashgari/issues/224
[#228]: https://github.com/BrikerMan/Kashgari/issues/228
[#244]: https://github.com/BrikerMan/Kashgari/issues/244
[#248]: https://github.com/BrikerMan/Kashgari/issues/248
[#249]: https://github.com/BrikerMan/Kashgari/issues/249