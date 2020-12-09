import os
import numpy as np
from typing import Union, Any, AnyStr

import tensorflow as tf
from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.logger import logger


class KashgariModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
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
                 filepath: AnyStr,
                 monitor: str = 'val_loss',
                 verbose: int = 1,
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = 'auto',
                 save_freq: Union[str, int] = 'epoch',
                 kash_model: ABCTaskModel = None,
                 **kwargs: Any) -> None:
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

    def _save_model(self, epoch: int, logs: dict) -> None:
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save: int = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.warning('Can save best model only with %s available, skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best: float = current
                        if self.save_weights_only:
                            filepath = os.path.join(filepath, 'cp')
                            self.model.save_weights(filepath, overwrite=True)
                            logger.info(f'checkpoint saved to {filepath}')
                        else:
                            self.kash_model.save(filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    filepath = os.path.join(filepath, 'cp')
                    self.model.save_weights(filepath, overwrite=True)
                    logger.info(f'checkpoint saved to {filepath}')
                else:
                    self.kash_model.save(filepath)

            self._maybe_remove_file()
