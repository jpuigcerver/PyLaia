from __future__ import absolute_import

from laia.engine.engine import Engine
from laia.utils import check_inf, check_nan
import logging
import numpy as np


class Trainer(Engine):
    r"""Wrapper class to train a model.

    See :class:`laia.engine.Engine` for more information.

    Arguments:
      model: model to train.
      data_loader (iterable): iterable object from which batches are read.
      batch_input_fn (callable): function used to extract the input for the
          model (e.g. a ``torch.Tensor``), from the batch loaded by the
          ``data_loader``. Use ``None`` to fed the batch as-is.
      batch_target_fn (callable): function used to extract the target passed
          to the ``criterion`` with the model output, from the batch loaded
          by the ``data_loader``. Use ``None`` to fed the batch as-is.
      criterion (callable): used criterion to train the model.
      optimizer (:class:`torch.Optimizer`): optimizer object that will update
          the parameters of the model.
      early_stop_trigger (callable, optional): function used to decide whether
          the training must stop (when the trigger returns ``True``) or
          continue (when return ``False``). If ``None``, the training will
          run forever. (default: None)
      progress_bar (bool or str, optional): if ``True``, :mod:`tqdm` will be
          used to show a progress bar for each epoch. If a string is given,
          the content of the string will be shown before the progress bar.
          If the module :mod:`tqdm` is not installed, this will be ignored.
          (default: None)
      num_iterations_to_update (int): Number of successive mini-batch
          parameter gradients to accumulate before updating the parameters.
          (default: None)
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, model, data_loader, criterion, optimizer,
                 batch_input_fn=None, batch_target_fn=None,
                 early_stop_trigger=None, progress_bar=None,
                 num_iterations_to_update=None):
        super(Trainer, self).__init__(model=model,
                                      data_loader=data_loader,
                                      batch_input_fn=batch_input_fn,
                                      batch_target_fn=batch_target_fn,
                                      progress_bar=progress_bar)
        self._criterion = criterion
        self._optimizer = optimizer
        self._early_stop_trigger = early_stop_trigger
        self._num_iterations_to_update = num_iterations_to_update

    @property
    def logger(self):
        return self._logger

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    def add_evaluator(self, evaluator):
        r"""Add an evaluator to run at the end of each epoch."""
        def run_eval(**_):
            evaluator.run()

        if evaluator is not None:
            self.add_hook(self.ON_EPOCH_END, run_eval)
        return self

    def set_criterion(self, criterion):
        assert callable(criterion)
        self._criterion = criterion
        return self

    def set_num_iterations_to_update(self, num):
        self._num_iterations_to_update = num
        return self

    def set_early_stop_trigger(self, trigger):
        r"""Set the trigger used for early stopping.

        Args:
          trigger (callable): a function or callable object that returns
              `True` when training must be stopped, or `False` otherwise.
        """
        assert trigger is None or callable(trigger)
        self._early_stop_trigger = trigger
        return self

    def run(self):
        r"""Run training until the early stop trigger returns True."""
        assert callable(self.criterion)
        assert callable(self._batch_input_fn), (
            'batch_input_fn (type: {!r}) is not callable'.format(
                str(self._batch_target_fn)))
        assert callable(self._batch_target_fn), (
            'batch_target_fn (type: {!r}) is not callable'.format(
                str(self._batch_target_fn)))

        while (self._early_stop_trigger is None or
               not self._early_stop_trigger()):
            self._run_epoch()
        return self

    def _run_iteration(self, it, batch):
        self._iterations += 1

        # Put model in training mode
        if hasattr(self._model, 'train'):
            self._model.train()

        # Prepare input to the model.
        if self._batch_input_fn is None:
            batch_input = batch
        else:
            batch_input = self._batch_input_fn(batch)

        # Prepare target to be passed to the loss function.
        if self._batch_target_fn is None:
            batch_target = batch
        else:
            batch_target = self._batch_target_fn(batch)

        self._call_hooks(self.ON_BATCH_START,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_target=batch_target)

        # Make all parameter gradients equal to zero.
        if (self._num_iterations_to_update is None or
                (it - 1) % self._num_iterations_to_update == 0):
            self._optimizer.zero_grad()

        # Run model, evaluate loss and compute gradients.
        batch_output = self._model(batch_input)

        # Note: These checks are only active when logging level >= DEBUG
        check_inf(tensor=batch_output, logger=self.logger,
                  msg='Found {abs_num} ({rel_num:.2%}) INF values in the '
                  'model output at epoch {epoch}, batch {batch} (absolute '
                  'iteration {iteration})',
                  epoch=self.epochs, batch=it, iteration=self.iterations)
        check_nan(tensor=batch_output, logger=self.logger,
                  msg='Found {abs_num} ({rel_num:.2%}) NAN values in the '
                  'model output at epoch {epoch}, batch {batch} (absolute '
                  'iteration {iteration})',
                  epoch=self.epochs, batch=it, iteration=self.iterations)

        batch_loss = self._criterion(batch_output, batch_target)
        batch_loss.backward()

        # Update model parameters.
        if (self._num_iterations_to_update is None or
                it % self._num_iterations_to_update == 0 or
                it == len(self._data_loader)):
            self.logger.debug(
                'Updating parameters at epoch {}, batch {} '
                '(absolute iteration {})'.format(
                    self.epochs, it, self.iterations))
            self._optimizer.step()

        self._call_hooks(self.ON_BATCH_END,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_target=batch_target,
                         batch_loss=batch_loss,
                         batch_output=batch_output)
