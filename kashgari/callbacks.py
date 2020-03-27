# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: callbacks.py
# time: 2019-05-22 15:00

import logging
import os

from seqeval import metrics as seq_metrics
from sklearn import metrics
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from kashgari import macros
from kashgari.tasks.base_model import BaseModel


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, kash_model: BaseModel, valid_x, valid_y,
                 step=5, batch_size=256, average='weighted'):
        """
        Evaluate callback, calculate precision, recall and f1
        Args:
            kash_model: the kashgari model to evaluate
            valid_x: feature data
            valid_y: label data
            step: step, default 5
            batch_size: batch size, default 256
        """
        super(EvalCallBack, self).__init__()
        self.kash_model = kash_model
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.step = step
        self.batch_size = batch_size
        self.average = average
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step == 0:
            y_pred = self.kash_model.predict(self.valid_x, batch_size=self.batch_size)

            if self.kash_model.task == macros.TaskType.LABELING:
                y_true = [seq[:len(y_pred[index])] for index, seq in enumerate(self.valid_y)]
                precision = seq_metrics.precision_score(y_true, y_pred)
                recall = seq_metrics.recall_score(y_true, y_pred)
                f1 = seq_metrics.f1_score(y_true, y_pred)
            else:
                y_true = self.valid_y
                precision = metrics.precision_score(y_true, y_pred, average=self.average)
                recall = metrics.recall_score(y_true, y_pred, average=self.average)
                f1 = metrics.f1_score(y_true, y_pred, average=self.average)

            self.logs.append({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            print(f"\nepoch: {epoch} precision: {precision:.6f}, recall: {recall:.6f}, f1: {f1:.6f}")


class KashgariModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """Save the model after every epoch.

     Arguments:
         filepath: string, path to save the model file.
         monitor: quantity to monitor.
         verbose: verbosity mode, 0 or 1.
         save_best_only: if `save_best_only=True`, the latest best model according
           to the quantity monitored will not be overwritten.
         mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
           overwrite the current save file is made based on either the maximization
           or the minimization of the monitored quantity. For `val_acc`, this
           should be `max`, for `val_loss` this should be `min`, etc. In `auto`
           mode, the direction is automatically inferred from the name of the
           monitored quantity.
         save_weights_only: if True, then only the model's weights will be saved
           (`model.save_weights(filepath)`), else the full model is saved
           (`model.save(filepath)`).
         save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
           the model after each epoch. When using integer, the callback saves the
           model at end of a batch at which this many samples have been seen since
           last saving. Note that if the saving isn't aligned to epochs, the
           monitored metric may potentially be less reliable (it could reflect as
           little as 1 batch, since the metrics get reset every epoch). Defaults to
           `'epoch'`
         **kwargs: Additional arguments for backwards compatibility. Possible key
           is `period`.
     """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 kash_model: BaseModel = None,
                 **kwargs):
        super(KashgariModelCheckpoint, self).__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            **kwargs)
        self.kash_model = kash_model

    def _save_model(self, epoch, logs):
        """Saves the model.

            Arguments:
                epoch: the epoch this iteration is in.
                logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
            """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            file_handle, filepath = self._get_file_handle_and_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            filepath = os.path.join(filepath, 'cp')
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.kash_model.save(filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    if K.in_multi_worker_mode():
                        # TODO(rchao): Save to an additional training state file for FT,
                        # instead of adding an attr to weight file. With this we can support
                        # the cases of all combinations with `save_weights_only`,
                        # `save_best_only`, and `save_format` parameters.
                        # pylint: disable=protected-access
                        self.model._ckpt_saved_epoch = epoch
                    filepath = os.path.join(filepath, 'cp')
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.kash_model.save(filepath)

            self._maybe_remove_file(file_handle, filepath)


if __name__ == "__main__":
    print("Hello world")
