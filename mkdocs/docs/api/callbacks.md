# callbacks

## class EvalCallBack

### \_\_init\_\_

Evaluate callback, calculate precision, recall and f1 at the end of each epoch step.

```python
def __init__(self,
             kash_model: BaseModel,
             valid_x,
             valid_y,
             step=5,
             batch_size=256):
```

__Args__:

- **kash_model**: the kashgari model to evaluate
- **valid_x**: feature data for evaluation
- **valid_y**: label data for evaluation
- **step**: evaluate step, default 5
- **batch_size**: batch size, default 256

### Methods

#### on\_epoch\_end

```python
def on_epoch_end(self, epoch, logs=None):
```
