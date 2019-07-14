# Labeling Models

All Text labeling models share the same API.

## \_\_init\_\_

```python
def __init__(self,
             embedding: Optional[Embedding] = None,
             hyper_parameters: Optional[Dict[str, Dict[str, Any]]] = None)
```

__Args__:

- **embedding**: model embedding
- **hyper_parameters**: a dict of hyper_parameters.

You could change customize hyper_parameters like this::

```python
# get default hyper_parameters
hyper_parameters = BiLSTM_Model.get_default_hyper_parameters()
# change lstm hidden unit to 12
hyper_parameters['layer_blstm']['units'] = 12
# init new model with customized hyper_parameters
labeling_model = BiLSTM_Model(hyper_parameters=hyper_parameters)
labeling_model.fit(x, y)
```

## build\_model\_arc

```python
def build_model_arc(self):
```

## build_model

build model with corpus

```python
def build_model(self,
                x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                y_train: Union[List[List[str]], List[str]],
                x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                y_validate: Union[List[List[str]], List[str]] = None)
```

__Args__:

- **x_train**: Array of train feature data (if the model has a single input),
      or tuple of train feature data array (if the model has multiple inputs)
- **y_train**: Array of train label data
- **x_validate**: Array of validation feature data (if the model has a single input),
      or tuple of validation feature data array (if the model has multiple inputs)
- **y_validate**: Array of validation label data

## build\_multi\_gpu\_model

Build multi-GPU model with corpus

```python
def build_multi_gpu_model(self,
                            gpus: int,
                            x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                            y_train: Union[List[List[str]], List[str]],
                            cpu_merge: bool = True,
                            cpu_relocation: bool = False,
                            x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                            y_validate: Union[List[List[str]], List[str]] = None):
```

__Args__:

- **gpus**: Integer >= 2, number of on GPUs on which to create model replicas.
- **cpu_merge**: A boolean value to identify whether to force merging model weights
    under the scope of the CPU or not.
- **cpu_relocation**: A boolean value to identify whether to create the model's weights
    under the scope of the CPU. If the model is not defined under any preceding device
    scope, you can still rescue it by activating this option.
- **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)
- **y_train**: Array of train label data
- **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)
- **y_validate**: Array of validation label data

## build\_tpu\_model

Build TPU model with corpus

```python
def build_tpu_model(self, strategy: tf.contrib.distribute.TPUStrategy,
                    x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                    y_train: Union[List[List[str]], List[str]],
                    x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                    y_validate: Union[List[List[str]], List[str]] = None):
```

__Args__:

- **strategy**: `TPUDistributionStrategy`. The strategy to use for replicating model
    across multiple TPU cores.
- **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)
- **y_train**: Array of train label data
- **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)
- **y_validate**: Array of validation label data

## compile_model

Configures the model for training.

Using `compile()` function of [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#compile)

```python
def compile_model(self, **kwargs):
```

__Args__:

- **\*\*kwargs**: arguments passed to `compile()` function of `tf.keras.Model`

__Defaults__:

- **loss**: ``categorical_crossentropy``
- **optimizer**: ``adam``
- **metrics**: ``['accuracy']``


## get\_data\_generator

data generator for fit_generator

```python
def get_data_generator(self,
                        x_data,
                        y_data,
                        batch_size: int = 64,
                        shuffle: bool = True)
```

__Args__:

- **x_data**: Array of feature data (if the model has a single input),
    or tuple of feature data array (if the model has multiple inputs)
- **y_data**: Array of label data
- **batch_size**: Number of samples per gradient update, default to 64.
- **shuffle**:

__Returns__:

- data generator

## fit

Trains the model for a given number of epochs with fit_generator (iterations on a dataset).

```python
def fit(self,
        x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
        y_train: Union[List[List[str]], List[str]],
        x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
        y_validate: Union[List[List[str]], List[str]] = None,
        batch_size: int = 64,
        epochs: int = 5,
        callbacks: List[keras.callbacks.Callback] = None,
        fit_kwargs: Dict = None):
```

__Args__:

- **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)
- **y_train**: Array of train label data
- **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)
- **y_validate**: Array of validation label data
- **batch_size**: Number of samples per gradient update, default to 64.
- **epochs**: Integer. Number of epochs to train the model. default 5.
- **callbacks**:
- **fit_kwargs**: additional arguments passed to `fit_generator()` function from
    [tensorflow.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator)

__Returns__:

- A `tf.keras.callbacks.History` object.

## fit\_without\_generator

Trains the model for a given number of epochs (iterations on a dataset). Large memory Cost.

```python
def fit_without_generator(self,
                            x_train: Union[Tuple[List[List[str]], ...], List[List[str]]],
                            y_train: Union[List[List[str]], List[str]],
                            x_validate: Union[Tuple[List[List[str]], ...], List[List[str]]] = None,
                            y_validate: Union[List[List[str]], List[str]] = None,
                            batch_size: int = 64,
                            epochs: int = 5,
                            callbacks: List[keras.callbacks.Callback] = None,
                            fit_kwargs: Dict = None):
```

__Args__:

- **x_train**: Array of train feature data (if the model has a single input),
    or tuple of train feature data array (if the model has multiple inputs)
- **y_train**: Array of train label data
- **x_validate**: Array of validation feature data (if the model has a single input),
    or tuple of validation feature data array (if the model has multiple inputs)
- **y_validate**: Array of validation label data
- **batch_size**: Number of samples per gradient update, default to 64.
- **epochs**: Integer. Number of epochs to train the model. default 5.
- **callbacks**:
- **fit_kwargs**: additional arguments passed to `fit_generator()` function from
    [tensorflow.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit_generator)

__Returns__:

- A `tf.keras.callbacks.History` object.

## predict

Generates output predictions for the input samples. Computation is done in batches.

```python
def predict(self,
            x_data,
            batch_size=32,
            debug_info=False):
```

__Args__:

- **x_data**: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
- **batch_size**: Integer. If unspecified, it will default to 32.
- **debug_info**: Bool, Should print out the logging info.

__Returns__:

- array of predictions.

## predict_entities

Gets entities from sequence.

```python
def predict_entities(self,
                     x_data,
                     batch_size=None,
                     join_chunk=' ',
                     debug_info=False):

```

__Args__:

- x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
- batch_size: Integer. If unspecified, it will default to 32.
- join_chunk: str or False,
- debug_info: Bool, Should print out the logging info.

__Returns__:

- list: list of entity.

## evaluate

Evaluate model

```python
def evaluate(self,
            x_data,
            y_data,
            batch_size=None,
            digits=4,
            debug_info=False) -> Tuple[float, float, Dict]:
```

__Args__:

- **x_data**:
- **y_data**:
- **batch_size**:
- **digits**:
- **debug_info**:

## save

Save model info json and model weights to given folder path

```python
def save(self, model_path: str):
```

__Args__:

- **model_path**: target model folder path
