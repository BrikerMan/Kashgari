# Tensorflow Serving

```python
from kashgari.tasks.classification import BiGRU_Model
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari import utils

train_x, train_y = SMP2018ECDTCorpus.load_data()

model = BiGRU_Model()
model.fit(train_x, train_y)

# Save model
utils.convert_to_saved_model(model,
                             model_path="saved_model/bgru",
                             version=1)
```

Then run tensorflow-serving.

```bash
docker run -t --rm -p 8501:8501 -v "<path_to>/saved_model:/models/" -e MODEL_NAME=bgru tensorflow/serving
```

Load processor from model, then predict with serving.

We need to check model input keys first.

```python
import requests
res = requests.get("http://localhost:8501/v1/models/bgru/metadata")
inputs = res.json()['metadata']['signature_def']['signature_def']['serving_default']['inputs']
input_sample_keys = list(inputs.keys())
print(input_sample_keys)
# ['Input-Token', 'Input-Segment']
```

If we have only one input key, aka we are not using BERT like embedding,
 we need to pass json in this format to predict endpoint.

```json
{
  "instances": [
    [2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 9, 41, 459, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ]
}
```

Here is the code.

```python
import requests
import numpy as np
from kashgari.processors import load_processors_from_model

text_processor, label_processor = load_processors_from_model('/Users/brikerman/Desktop/tf-serving/1603683152')

samples = [
    ['hello', 'world'],
    ['你', '好', '世', '界']
]
tensor = text_processor.transform(samples)

instances = [i.tolist() for i in tensor]

# predict
r = requests.post("http://localhost:8501/v1/models/bgru:predict", json={"instances": instances})
predictions = r.json()['predictions']

# Convert result back to labels
labels = label_processor.inverse_transform(np.array(predictions).argmax(-1))
print(labels)
```

If we are using Bert, then we need to handle multi input fields,
 for example we get this two keys `['Input-Token', 'Input-Segment']` from metadata endpoint.
 Then we need to pass a json in this format.

```json
[
  {
    "Input-Token": [2, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Input-Segment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  },
  {
    "Input-Token": [2, 9, 41, 459, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Input-Segment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  }
]

```

Here is the code.

```python
import requests
import numpy as np
from kashgari.processors import load_processors_from_model

text_processor, label_processor = load_processors_from_model('/Users/brikerman/Desktop/tf-serving/1603683152')

samples = [
    ['hello', 'world'],
    ['你', '好', '世', '界']
]
tensor = text_processor.transform(samples)

instances = [{
   "Input-Token": i.tolist(),
   "Input-Segment": np.zeros(i.shape).tolist()
} for i in tensor]

# predict
r = requests.post("http://localhost:8501/v1/models/bgru:predict", json={"instances": instances})
predictions = r.json()['predictions']

# Convert result back to labels
labels = label_processor.inverse_transform(np.array(predictions).argmax(-1))
print(labels)
```
