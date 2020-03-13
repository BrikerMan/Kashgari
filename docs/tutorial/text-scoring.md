# Text Scoring Model

Kashgari provides several models for text scoring, which could be use for Sentiment analysis tasks. Model input is text and output is continuous float value.
All labeling models inherit from the `BaseScoringModel`.
You could easily switch from one model to another just by changing one line of code.

## Available Models

| Name                  | info |
| --------------------- | ---- |
| BiLSTM\_Model         |      |

## Train basic scoring model



```python
# Load build-in corpus.
from kashgari.corpus import SMP2018ECDTCorpus

# Sample x is tokenized text, y is float value
train_x = [['Hello', 'world'], ['Hello', 'Kashgari'], ['I', 'hate', 'you']]
train_y = [5.0, 5.0, 1.2]

valid_x, valid_y = train_x, train_y
test_x, test_x = train_x, train_y
```

Then train our first model. All models provided some APIs, so you could use any scoring model here.

```python
import kashgari
from kashgari.tasks.scoring import BiLSTM_Model

import logging
logging.basicConfig(level='DEBUG')

model = BiLSTM_Model()
model.fit(train_x, train_y, valid_x, valid_y)

# Evaluate the model
model.evaluate(test_x, test_y)

# Evaluate the model with round funcion
model.evaluate(test_x, test_y, should_round=True)

# Model data will save to `saved_scoring_model` folder
model.save('saved_scoring_model')

# Load saved model
loaded_model = kashgari.utils.load_model('saved_scoring_model')
loaded_model.predict(test_x[:10])

# To continue training, compile the newly loaded model first
loaded_model.compile_model()
model.fit(train_x, train_y, valid_x, valid_y)
```

That's all your need to do. Easy right.

## Text scoring with transfer learning

Kashgari provides varies Language model Embeddings for transfer learning. Here is the example for BERT Embedding.

```python
import kashgari
from kashgari.tasks.scoring import BiGRU_Model
from kashgari.embeddings import BERTEmbedding

import logging
logging.basicConfig(level='DEBUG')

bert_embed = BERTEmbedding('<PRE_TRAINED_BERT_MODEL_FOLDER>',
                           task=kashgari.SCORING,
                           sequence_length=100)
model = BiGRU_Model(bert_embed)
model.fit(train_x, train_y, valid_x, valid_y)
```

You could replace bert_embedding with any Embedding class in `kashgari.embeddings`. More info about Embedding: LINK THIS.

## Adjust model's hyper-parameters

You could easily change model's hyper-parameters. For example, we change the lstm unit in `BiLSTM_Model` from 128 to 32.

```python
from kashgari.tasks.scoring import BiLSTM_Model

hyper = BiLSTM_Model.get_default_hyper_parameters()
print(hyper)
# {'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_dense': {'activation': 'softmax'}}

hyper['layer_bi_lstm']['units'] = 32

model = BiLSTM_Model(hyper_parameters=hyper)
```

## Use custom optimizer

Kashgari already supports using customized optimizer, like RAdam.

```python
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.tasks.scoring import BiLSTM_Model
# Remember to import kashgari before than RAdam
from keras_radam import RAdam

model = BiLSTM_Model()
# This step will build token dict, label dict and model structure
model.build_model(train_x, train_y, valid_x, valid_y)
# Compile model with custom optimizer, you can also customize loss and metrics.
optimizer = RAdam()
model.compile_model(optimizer=optimizer)

# Train model 
model.fit(train_x, train_y, valid_x, valid_y)
```

## Use callbacks

Kashgari is based on keras so that you could use all of the [tf.keras callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) directly with
Kashgari model. For example, here is how to visualize training with tensorboard.

```python
from tensorflow.python import keras
from kashgari.tasks.scoring import BiGRU_Model
from kashgari.callbacks import EvalCallBack

import logging
logging.basicConfig(level='DEBUG')

model = BiGRU_Model()

tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          batch_size=100,
          callbacks=[tf_board_callback])
```

## Customize your own model

It is very easy and straightforward to build your own customized model,
just inherit the `BaseScoringModel` and implement the `get_default_hyper_parameters()` function and `build_model_arc()` function.

```python
from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.scoring.base_model import BaseScoringModel
from kashgari.layers import L

import logging
logging.basicConfig(level='DEBUG')


class DoubleBiLSTMModel(BaseScoringModel):
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
                'activation': 'linear'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = self.processor.output_dim
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

## Speed up with CuDNN cell

You can speed up training and inferencing process using [CuDNN cell](https://stackoverflow.com/questions/46767001/what-is-cudnn-implementation-of-rnn-cells-in-tensorflow). CuDNNLSTM and CuDNNGRU layers are much faster than LSTM and GRU layer, but they must be used on GPU. If you want to train on GPU and inferencing on CPU, you cannot use CuDNN cells.

```python
# Enable use cudnn cell
kashgari.config.use_cudnn_cell = True
```
