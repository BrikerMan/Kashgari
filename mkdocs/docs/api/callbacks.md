# callbacks

## class EvalCallBack

Evaluate callback, calculate precision, recall and f1 at the end of each epoch step.

__Args__:

- **kash_model**: the kashgari model to evaluate
- **valid_x**: feature data for evaluation
- **valid_y**: label data for evaluation
- **step**: evaluate step, default 5
- **batch_size**: batch size, default 256
