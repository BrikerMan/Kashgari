# FAQ

## How can I run Keras on GPU

Kashgari will use GPU by default if available, but you need to setup the Tensorflow GPU environment first. You can check gpu status using the code below:

```python
import tensorflow as tf
print(tf.test.is_gpu_available())
```

Here is the official document of [TensorFlow-GPU](https://www.tensorflow.org/install/gpu)

## How to save and resume training with ModelCheckpoint callback

You can use [tf.keras.callbacks.ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) for saving model during training.

```python
from tensorflow.python.keras.callbacks import ModelCheckpoint

filepath = "saved-model-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint_callback = ModelCheckpoint(filepath,
                                      monitor = 'acc',
                                      verbose = 1)

model = CNN_GRU_Model()
model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          callbacks=[checkpoint_callback])
```

ModelCheckpoint will save models struct and weights to target file, but we need token dict and label dict to fully restore the model, so we have to save model using `model.save()` function.

So, the full solution will be like this.

```python
from tensorflow.python.keras.callbacks import ModelCheckpoint

filepath = "saved-model-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint_callback = ModelCheckpoint(filepath,
                                      monitor = 'acc',
                                      verbose = 1)

model = CNN_GRU_Model()

# This function will build token dict, label dict and model struct.
model.build_model(train_x, train_y, valid_x, valid_y)
# Save full model info and initial weights to the full_model folder.
model.save('full_model')

# Start Training
model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          callbacks=[checkpoint_callback])


# Load Model
from kashgari.utils import load_model

# We only need model struct and dicts
new_model = load_model('full_model', load_weights=False)
# Load weights from ModelCheckpoint
new_model.tf_model.load_weights('saved-model-05-0.96.hdf5')

# Resume Training
# Only need to set {'initial_epoch': 5} when you wish to start new epoch from 6
# Otherwise epoch will start from 1
model.fit(train_x,
          train_y,
          valid_x,
          valid_y,
          callbacks=[checkpoint_callback],
          epochs=10,
          fit_kwargs={'initial_epoch': 5})
```
