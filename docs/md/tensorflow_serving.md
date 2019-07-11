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
                             model_path='saved_model/bgru', 
                             version=1)
```

Then run tensorflow-serving.

```bash
docker run -t --rm -p 8501:8501 -v "path_to/saved_model:/models/" -e MODEL_NAME=bgru tensorflow/serving
```

Load processor from model, then predict with serving.

```python
import requests
from kashgari import utils
import numpy as np

x = ['Hello', 'World']
# Pre-processor data
processor = utils.load_processor(model_path='saved_model/bgru/1')
tensor = processor.process_x_dataset([x])

# array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)

# predict
r = requests.post("http://localhost:8501/v1/models/bgru:predict", json={"instances": tensor.tolist()})
preds = r.json()['predictions']

# Convert result back to labels
labels = processor.reverse_numerize_label_sequences(np.array(preds).argmax(-1))

# labels = ['video']
```