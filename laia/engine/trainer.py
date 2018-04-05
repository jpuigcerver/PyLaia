from __future__ import absolute_import

import laia.logging as log
from laia.engine.engine import Engine, ON_BATCH_START, ON_BATCH_END, ON_EPOCH_END
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
          passed to the `ON_BATCH_START` and `ON_BATCH_END` hooks.
      criterion (callable): used criterion to train the model.
      optimizer (:class:`torch.Optimizer`): optimizer object that will update
          the parameters of the model.
      progress_bar (bool or str): if ``True``, :mod:`tqdm` will be
          used to show a progress bar for each epoch. If a string is given,
          the content of the string will be shown before the progress bar.
          If the module :mod:`tqdm` is not installed, this will be ignored.
          (default: None)
      num_iterations_per_update (int): Number of successive mini-batch
          parameter gradients to accumulate before updating the parameters.
          (default: None)
    """

    def __init__(self, model, data_loader, criterion, optimizer,
                 batch_input_fn=None, batch_target_fn=None,
                 progress_bar=None, num_iterations_per_update=None):
        super(Trainer, self).__init__(model=model,
                                      data_loader=data_loader,
                                      batch_input_fn=batch_input_fn,
                                      batch_target_fn=batch_target_fn,
                                      progress_bar=progress_bar)
        self._criterion = criterion
        self._optimizer = optimizer
        self._num_iterations_per_update = None
        self._updates = 0

        # Initialize _num_iterations_per_update
        self.set_num_iterations_per_update(num_iterations_per_update)

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def updates(self):
        return self._updates

    @property
    def logger(self):
        return _logger

    def add_evaluator(self, evaluator):
        r"""Add an evaluator to run at the end of each epoch."""

        def run_eval(**_):
            evaluator.run()

        if evaluator is not None:
            self.add_hook(ON_EPOCH_END, run_eval)
        return self

    def set_criterion(self, criterion):
        assert callable(criterion)
        self._criterion = criterion
        return self

    def set_num_iterations_per_update(self, num):
        if num is None:
            self._num_iterations_per_update = 1
        else:
            assert isinstance(num, int)
            assert num > 0
            self._num_iterations_per_update = num
        return self

    def _run_iteration(self, it, batch):
        self._iterations += 1

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

        self._call_hooks(ON_BATCH_START,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_target=batch_target)

        # Make all parameter gradients equal to zero.
        # Note: (IT - 1) % NIPU = the iteration after a step()
        if (self.iterations() - 1) % self._num_iterations_per_update == 0:
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
                  epoch=self.epochs(), batch=it, iteration=self.iterations())
        check_nan(tensor=batch_output, logger=__name__,
                  msg='Found {abs_num} ({rel_num:.2%}) NAN values in the '
                      'model output at epoch {epoch}, batch {batch} (absolute '
                      'iteration {iteration})',
                  epoch=self.epochs(), batch=it, iteration=self.iterations())
        # Compute loss
        batch_loss = self._criterion(batch_output, batch_target)

        # Make the loss and gradients w.r.t. output independent of the number
        # of accumulated iterations.
        if self._num_iterations_per_update > 1:
            batch_loss /= self._num_iterations_per_update

        # Compute gradients w.r.t. parameters
        self.logger.debug('Start backward at epoch {}, batch {} '
                          '(absolute iteration {})',
                          self.epochs(), it, self.iterations())
        batch_loss.backward()

        # Update model parameters.
        if self.iterations() % self._num_iterations_per_update == 0:
            self._updates += 1
            self.logger.debug('Updating parameters at epoch {}, batch {} '
                              '(absolute iteration {})',
                              self.epochs(), it, self.iterations())
            self._optimizer.step()

        self._call_hooks(ON_BATCH_END,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_target=batch_target,
                         batch_loss=batch_loss,
                         batch_output=batch_output)

    def state_dict(self):
        engine_state = super(Trainer, self).state_dict()
        return {
            # TODO
            'engine_state': engine_state
        }

    def load_state_dict(self, state):
        super(Trainer, self).load_state_dict(state['engine_state'])
        # TODO
