from __future__ import absolute_import

from laia.engine.engine import Engine


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
    """
    def __init__(self, model, data_loader, batch_input_fn, batch_target_fn,
                 criterion, optimizer, early_stop_trigger=None,
                 progress_bar=None):
        super(Trainer, self).__init__(model=model,
                                      data_loader=data_loader,
                                      batch_input_fn=batch_input_fn,
                                      batch_target_fn=batch_target_fn,
                                      progress_bar=progress_bar)
        self._criterion = criterion
        self._optimizer = optimizer
        self._early_stop_trigger = None
        self.set_early_stop_trigger(early_stop_trigger)

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    def add_evaluator(self, evaluator):
        r"""Add an evaluator to run at the end of each epoch."""
        def run_eval(**kwargs):
            evaluator.run()

        if evaluator is not None:
            self.add_hook(self.ON_EPOCH_END, run_eval)
        return self

    def set_batch_target_fn(self, fn):
        r"""Set the function to obtain the targets for the loss.

        The argument can be either a function or a callable object that
        will receive as a single argument the batch read from the
        ``data_loader``, and must return the appropriate target for the
        used loss to train the model.
        """
        if fn is None:
            self._batch_target_fn = lambda x: x
        else:
            assert(callable(fn))
            self._batch_target_fn = fn
        return self

    def set_criterion(self, criterion):
        assert callable(criterion)
        self._criterion = criterion
        return self

    def set_early_stop_trigger(self, trigger):
        r"""Set the trigger used for early stopping.

        Args:
          trigger (callable): a function or callable object that returns
              True when training must be stopped, or False otherwise.
        """
        if trigger is None:
            self._early_stop_trigger = lambda: False
        else:
            assert(callable(trigger))
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

        while not self._early_stop_trigger():
            self._run_epoch()
        return self

    def _run_iteration(self, it, batch):
        self._iterations += 1

        batch_input = self._batch_input_fn(batch)
        batch_target = self._batch_target_fn(batch)

        self._call_hooks(self.ON_BATCH_START,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_target=batch_target)


        loss_and_output = [None, None]
        def closure():
            if hasattr(self._model, 'eval'):
                self._model.train()
            batch_output = self._model(batch_input)
            batch_loss = self._criterion(batch_output, batch_target)
            batch_loss.backward()
            loss_and_output[0] = batch_loss
            loss_and_output[1] = batch_output
            return batch_loss

        self._optimizer.zero_grad()
        self._optimizer.step(closure)

        self._call_hooks(self.ON_BATCH_END,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_target=batch_target,
                         batch_loss=loss_and_output[0],
                         batch_output=loss_and_output[1])
