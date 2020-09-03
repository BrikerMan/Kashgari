# Performance

This is for run performance report on models with bert-embedding.


## Classification

```python
from kashgari.corpus import SMP2018ECDTCorpus

train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')
```

|    | model_name          |   epoch |   f1-score |   precision |   recall | time   |
|---:|:--------------------|--------:|-----------:|------------:|---------:|:-------|
|  0 | BiGRU_Model         |      10 |   0.9335   |    0.937795 | 0.935065 | 00:33  |
|  1 | BiLSTM_Model        |      10 |   0.929075 |    0.930548 | 0.92987  | 00:33  |
|  2 | CNN_Attention_Model |      10 |   0.862197 |    0.888507 | 0.866234 | 00:27  |
|  3 | CNN_GRU_Model       |      10 |   0.840024 |    0.886519 | 0.850649 | 00:28  |
|  4 | CNN_LSTM_Model      |      10 |   0.424649 |    0.551247 | 0.511688 | 00:27  |
|  5 | CNN_Model           |      10 |   0.930336 |    0.938373 | 0.931169 | 00:26  |

## NER

```python
from kashgari.corpus import ChineseDailyNerCorpus

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')
```

|    | model_name       |   epoch |   f1-score |   precision |   recall | time   |
|---:|:-----------------|--------:|-----------:|------------:|---------:|:-------|
|  0 | BiGRU_Model      |      10 |   0.921583 |    0.913184 | 0.930532 | 19:10  |
|  1 | BiGRU_CRF_Model  |      10 |   0.935163 |    0.931246 | 0.939118 | 24:30  |
|  2 | BiLSTM_Model     |      10 |   0.915363 |    0.906566 | 0.924418 | 19:12  |
|  3 | BiLSTM_CRF_Model |      10 |   0.940539 |    0.944549 | 0.936646 | 24:31  |
|  4 | CNN_LSTM_Model   |      10 |   0.919783 |    0.909695 | 0.930272 | 19:07  |
