# Text Classification

Kashgari provides `CNN_Model`, `CNN_LSTM_Model`, `BLSTM_Model` and `BLSTM_CRF_Model` for sequence labeling, All labeling models inherit from the `SequenceLabelingModel`. It is almost identical to the text classification class `ClassificationModel`. Except the data type for Y in classification model is `List[str]`, for labeling it is `List[List[str]]`

See more: [Tutorial 2: Text Classification](Tutorial_3_Sequence_Labeling.md)