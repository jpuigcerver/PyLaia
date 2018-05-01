from __future__ import absolute_import

import laia.logging as log
from laia.engine.engine import Engine, EPOCH_END, ITER_START, ITER_END
from laia.hooks import Hook, action
from laia.utils import check_inf, check_nan

_logger = log.get_logger(__name__)


class Trainer(Engine):
    r"""Wrapper class to train a model.

    See :class:`laia.engine.Engine` for more information.

    Arguments:
      model: model to train.
      data_loader (iterable): iterable object from which batches are read.
      batch_input_fn (callable, optional): function used to extract the input
          for the model (e.g. a ``torch.Tensor``), from the batch loaded by
          the ``data_loader``. If ``None``, the batch is fed as-is to the
          model. (default: None)
      batch_target_fn (callable, optional): if given, this callable object
          is used to extract the targets from the batch, which are
          passed to the `ITER_START` and `ITER_END` hooks.
      criterion (callable): used criterion to train the model.
      optimizer (:class:`torch.Optimizer`): optimizer object that will update
          the parameters of the model.
      progress_bar (bool or str): if ``True``, :mod:`tqdm` will be
          used to show a progress bar for each epoch. If a string is given,
          the content of the string will be shown before the progress bar.
          If the module :mod:`tqdm` is not installed, this will be ignored.
          (default: None)
      iterations_per_update (int): Number of successive mini-batch
          parameter gradients to accumulate before updating the parameters.
          (default: None)
    """

    def __init__(self, model, criterion, optimizer,
                 data_loader=None,
                 batch_input_fn=None,
                 batch_target_fn=None,
                 progress_bar=None,
                 iterations_per_update=1):
        super(Trainer, self).__init__(model=model,
                                      data_loader=data_loader,
                                      batch_input_fn=batch_input_fn,
                                      batch_target_fn=batch_target_fn,
                                      progress_bar=progress_bar)
        self._criterion = criterion
        self._optimizer = optimizer
        self._iterations_per_update = iterations_per_update
        self._updates = 0

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        assert callable(criterion)
        self._criterion = criterion

    @property
    def optimizer(self):
        return self._optimizer

    def updates(self):
        return self._updates

    @property
    def logger(self):
        return _logger

    @property
    def iterations_per_update(self):
        return self._iterations_per_update

    @iterations_per_update.setter
    def iterations_per_update(self, num):
        if num is None:
            self._iterations_per_update = 1
        else:
            assert isinstance(num, int)
            assert num > 0
            self._iterations_per_update = num

    def add_evaluator(self, evaluator, when=EPOCH_END, condition=None):
        r"""Add an evaluator to run at the end of each epoch."""
        if evaluator is not None:
            self.add_hook(
                when,
                Hook(condition, evaluator.run)
                if condition is not None else evaluator.run)
        return self

    @action
    def run(self):
        r"""Run training """
        assert callable(self.criterion)
        assert callable(self._batch_input_fn), (
            'batch_input_fn (type: {!r}) is not callable'.format(
                str(self._batch_target_fn)))
        assert callable(self._batch_target_fn), (
            'batch_target_fn (type: {!r}) is not callable'.format(
                str(self._batch_target_fn)))
        while not self._must_stop:
            self._run_epoch()
        return self

    def _run_iteration(self, batch_n, batch):
        batch_input, batch_target = self._prepare_input_and_target(batch)

        action_kwargs = {
            'batch': batch,
            'batch_num': batch_n,
            'epoch': self._epochs,
            'iteration': self._iterations,
            'batch_input': batch_input,
            'batch_target': batch_target}
        self._call_hooks(ITER_START, **action_kwargs)

        # Make all parameter gradients equal to zero.
        # Note: IT % NIPU = the iteration after a step()
        if self._iterations % self.iterations_per_update == 0:
            self._optimizer.zero_grad()

        # Put model in training mode
        if hasattr(self._model, 'train'):
            self._model.train()

        # Run model, evaluate loss and compute gradients.
        batch_output = self._model(batch_input)

        # Note: These checks are only active when logging level >= DEBUG
        check_inf(tensor=batch_output, logger=__name__,
                  msg='Found {abs_num} ({rel_num:.2%}) INF values in the '
                      'model output at epoch {epoch}, batch {batch} (absolute '
                      'iteration {iteration})',
                  epoch=self._epochs, batch=batch_n, iteration=self._iterations)
        check_nan(tensor=batch_output, logger=__name__,
                  msg='Found {abs_num} ({rel_num:.2%}) NAN values in the '
                      'model output at epoch {epoch}, batch {batch} (absolute '
                      'iteration {iteration})',
                  epoch=self._epochs, batch=batch_n, iteration=self._iterations)

        # Compute loss
        batch_loss = self._criterion(batch_output, batch_target)

        # Make the loss and gradients w.r.t. output independent of the number
        # of accumulated iterations.
        if self.iterations_per_update > 1:
            batch_loss /= self.iterations_per_update

        # Compute gradients w.r.t. parameters
        self.logger.debug('Start backward at epoch {}, batch {} '
                          '(absolute iteration {})',
                          self._epochs, batch_n, self._iterations)
        batch_loss.backward()

        self._iterations += 1

        # Update model parameters.
        if self._iterations % self.iterations_per_update == 0:
            self._updates += 1
            self.logger.debug('Updating parameters at epoch {}, batch {} '
                              '(absolute iteration {})',
                              self._epochs, batch_n, self._iterations)
            self._optimizer.step()

        action_kwargs['iterations'] = self._iterations
        action_kwargs['batch_output'] = batch_output
        action_kwargs['batch_loss'] = batch_loss
        self._call_hooks(ITER_END, **action_kwargs)

    def state_dict(self):
        return {
            'engine_state': super(Trainer, self).state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'updates': self.updates()}

    def load_state_dict(self, state):
        super(Trainer, self).load_state_dict(state['engine_state'])
        self._optimizer.load_state_dict(state['optimizer_state'])
        self._updates = state['updates']
