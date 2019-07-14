# Text Classification Model

Kashgari provides several models for text classification,
All labeling models inherit from the `BaseClassificationModel`.
You could easily switch from one model to another just by changing one line of code.

## Available Models

| Name                | info |
| ------------------- | ---- |
| BiLSTM_Model        |      |
| BiGRU_Model         |      |
| CNN_Model           |      |
| CNN_LSTM_Model      |      |
| CNN_GRU_Model       |      |
| AVCNN_Model         |      |
| KMax_CNN_Model      |      |
| R_CNN_Model         |      |
| AVRNN_Model         |      |
| Dropout_BiGRU_Model |      |
| Dropout_AVRNN_Model |      |
| DPCNN_Model         |      |

## Train basic classification model

Kashgari provices basic intent-classification corpus for expirement. You could also use your corpus in any language for training.

Load build-in corpus.

```python
from kashgari.corpus import SMP2018ECDTCorpus

train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')
```

Then train our first model. All models provided some APIs, so you could use any labeling model here.

```python
import kashgari
from kashgari.tasks.classification import BiLSTM_Model

import logging
logging.basicConfig(level='DEBUG')

model = BiLSTM_Model()
model.fit(train_x, train_y, valid_x, valid_y)

# Evaluate the model
model.evaluate(test_x, test_y)

# Model data will save to `saved_ner_model` folder
model.save('saved_classification_model')

# Load saved model
loaded_model = kashgari.utils.load_model('saved_classification_model')
loaded_model.predict(test_x[:10])
```

That's all your need to do. Easy right.

## Text classification with transfer learning

Kashgari provides varies Language model Embeddings for transfer learning. Here is the example for BERT Embedding.

```python
import kashgari
from kashgari.tasks.classification import BiGRU_Model
from kashgari.embeddings import BERTEmbedding

import logging
logging.basicConfig(level='DEBUG')

bert_embed = BERTEmbedding('<PRE_TRAINED_BERT_MODEL_FOLDER>',
                           task=kashgari.LABELING,
                           sequence_length=100)
model = BiGRU_Model(bert_embed)
model.fit(train_x, train_y, valid_x, valid_y)
```

You could replace bert_embedding with any Embedding class in `kashgari.embeddings`. More info about Embedding: LINK THIS.

## Adjust model's hyper-parameters

You could easily change model's hyper-parameters. For example, we change the lstm unit in `BiLSTM_Model` from 128 to 32.

```python
from kashgari.tasks.classification import BiLSTM_Model

hyper = BiLSTM_Model.get_default_hyper_parameters()
print(hyper)
# {'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_dense': {'activation': 'softmax'}}

hyper['layer_bi_lstm']['units'] = 32

model = BiLSTM_Model(hyper_parameters=hyper)
```

## Use callbacks

Kashgari is based on keras so that you could use all of the [tf.keras callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) directly with
Kashgari model. For example, here is how to visualize training with tensorboard.

```python
from tensorflow.python import keras
from kashgari.tasks.classification import BiGRU_Model
from kashgari.callbacks import EvalCallBack

import logging
logging.basicConfig(level='DEBUG')

model = BiGRU_Model()

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

# Build-in callback for print precision, recall and f1 at every epoch step
eval_callback = EvalCallBack(kash_model=model,
                             valid_x=valid_x,
                             valid_y=valid_y,
                             step=5)

model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          batch_size=100,
          callbacks=[eval_callback, tf_board_callback])
```

## Multi-Label Classification

Kashgari support multi-label classification, Here is how we build one.

Let's assume we have a dataset like this.

```python
x = [
   ['This','news',are',very','well','organized'],
   ['What','extremely','usefull','tv','show'],
   ['The','tv','presenter','were','very','well','dress'],
   ['Multi-class', 'classification', 'means', 'a', 'classification', 'task', 'with', 'more', 'than', 'two', 'classes']
]

y = [
   ['A', 'B'],
   ['A',],
   ['B', 'C'],
   []
]
```

Now we need to init a `Processor` and `Embedding` for our model, then prepare model and fit.

```python
from kashgari.tasks.classification import BiLSTM_Model
from kashgari.processors import ClassificationProcessor
from kashgari.embeddings import BareEmbedding

import logging
logging.basicConfig(level='DEBUG')

processor = ClassificationProcessor(multi_label=True)
embed = BareEmbedding(processor=processor)

model = BiLSTM_Model(embed)
model.fit(x, y)
```

## Customize your own model

It is very easy and straightforward to build your own customized model,
just inherit the `BaseClassificationModel` and implement the `get_default_hyper_parameters()` function
and `build_model_arc()` function.

```python
from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.base_model import BaseClassificationModel
from kashgari.layers import L

import logging
logging.basicConfig(level='DEBUG')


class DoubleBLSTMModel(BaseClassificationModel):
    """Bidirectional LSTM Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_blstm2': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define your layers
        layer_blstm1 = L.Bidirectional(L.LSTM(**config['layer_blstm1']),
                                       name='layer_blstm1')
        layer_blstm2 = L.Bidirectional(L.LSTM(**config['layer_blstm2']),
                                       name='layer_blstm2')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        # Define tensor flow
        tensor = layer_blstm1(embed_model.output)
        tensor = layer_blstm2(tensor)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        # Init model
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

model = DoubleBLSTMModel()
model.fit(train_x, train_y, valid_x, valid_y)
```
