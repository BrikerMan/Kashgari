# Customize Multi Output Model

It is very easy to customize your own multi output model. Lets assume you have dataset like this, One input and two output.

Example code at file `tests/test_custom_multi_output_classification.py`.

```python
x = [
    ['我', '想', '全', '部', '回', '复'], 
    ['我', '想', '上', 'q', 'q', '了'], 
    ['你', '去', '过', '赌', '场', '吗'], 
    ['是', '我', '是', '说', '你', '有', '几', '个', '兄', '弟', '姐', '妹', '不', '是', '你', '自', '己', '说'], 
    ['广', '西', '新', '闻', '网']
]

output_1 = [
    [0. 0. 1.]
    [1. 0. 0.]
    [1. 0. 0.]
    [0. 0. 1.]
    [1. 0. 0.]]

output_2 = [
    [0. 1. 0.]
    [0. 0. 1.]
    [0. 0. 1.]
    [1. 0. 0.]
    [0. 0. 1.]]
```

Then you need to create a customized processor inhered from the `ClassificationProcessor`.

```python
import kashgari
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from kashgari.processors.classification_processor import ClassificationProcessor

class MultiOutputProcessor(ClassificationProcessor):
    def process_y_dataset(self,
                          data: Tuple[List[List[str]], ...],
                          maxlens: Optional[Tuple[int, ...]] = None,
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        # Data already converted to one-hot
        # Only need to get the subset
        result = []
        for index, dataset in enumerate(data):
            if subset is not None:
                target = kashgari.utils.get_list_subset(dataset, subset)
            else:
                target = dataset
            result.append(np.array(target))

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)
```

Then build your own model inhered from the `BaseClassificationModel`

```python
import kashgari
import tensorflow as tf
from typing import Tuple, List, Optional, Dict, Any
from kashgari.layers import L
from kashgari.tasks.classification.base_model import BaseClassificationModel


class MultiOutputModel(BaseClassificationModel):
    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 256,
                'return_sequences': False
            }
        }

    # Build your own model
    def build_model_arc(self):
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']), name='layer_bi_lstm')
        layer_output_1 = L.Dense(3, activation='sigmoid', name='layer_output_1')
        layer_output_2 = L.Dense(3, activation='sigmoid', name='layer_output_2')

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor_1 = layer_output_1(tensor)
        output_tensor_2 = layer_output_2(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, [output_tensor_1, output_tensor_2])

    # Rewrite your predict function
    def predict(self,
                x_data,
                batch_size=None,
                debug_info=False,
                threshold=0.5):
        tensor = self.embedding.process_x_dataset(x_data)
        pred = self.tf_model.predict(tensor, batch_size=batch_size)

        output_1 = pred[0]
        output_2 = pred[1]

        output_1[output_1 >= threshold] = 1
        output_1[output_1 < threshold] = 0
        output_2[output_2 >= threshold] = 1
        output_2[output_2 < threshold] = 0

        return output_1, output_2
```

Tada, all done, Now build your own model with customized processor

```python
from kashgari.embeddings import BareEmbedding

# Use your processor to init embedding, You can use any embedding layer provided by kashgari here

processor = MultiOutputProcessor()
embedding = BareEmbedding(processor=processor)

m = MultiOutputModel(embedding=embedding)
m.build_model(train_x, (output_1, output_2))
m.fit(train_x, (output_1, output_2))
```