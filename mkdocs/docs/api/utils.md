# utils

## Methods

### unison\_shuffled\_copies

```python
def unison_shuffled_copies(a, b)
```

### get\_list\_subset

```python
def get_list_subset(target: List, index_list: List[int]) -> List
```

### custom\_object\_scope

```python
def custom_object_scope()
```

### load_model

Load saved model from saved model from `model.save` function

```python
def load_model(model_path: str, load_weights: bool = True) -> BaseModel
```

__Args__:

- model_path: model folder path
- load_weights: only load model structure and vocabulary when set to False, default True.

__Returns__:

### load_processor

```python
def load_processor(model_path: str) -> BaseProcessor
```

Load processor from model, When we using tf-serving, we need to use model's processor to pre-process data

__Args__:
    model_path:

__Returns__:

### convert\_to\_saved\_model

Export model for tensorflow serving

```python
def convert_to_saved_model(model: BaseModel,
                           model_path: str,
                           version: str = None,
                           inputs: Optional[Dict] = None,
                           outputs: Optional[Dict] = None):
```

__Args__:

- model: Target model
- model_path: The path to which the SavedModel will be stored.
- version: The model version code, default timestamp
- inputs: dict mapping string input names to tensors. These are added
    to the SignatureDef as the inputs.
- outputs:  dict mapping string output names to tensors. These are added
    to the SignatureDef as the outputs.
