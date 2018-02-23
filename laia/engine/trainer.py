from __future__ import absolute_import

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


class Trainer(object):
    def __init__(self, model, criterion, optimizer, dataset,
                 batch_input_fn=None, batch_target_fn=None,
                 early_stop_trigger=None,
                 epoch_saver_trigger=None,
                 iteration_saver_trigger=None):
        self._model = model
        self._criterion = criterion
        self._dataset = dataset
        self._optimizer = optimizer
        self._iterations = 0
        self._epochs = 0
        self._batch_input_fn = None
        self._batch_target_fn = None
        self._early_stop_trigger = None
        self._epoch_saver_trigger = None
        self._iteration_saver_trigger = None
        self._hooks = {
            'on_start_epoch': [],
            'on_start_batch': [],
            'on_end_epoch': [],
            'on_end_batch': [],
        }

        # Default functions
        self.set_batch_input_fn(batch_input_fn)
        self.set_batch_target_fn(batch_target_fn)
        self.set_early_stop_trigger(early_stop_trigger)
        self.set_epoch_saver_trigger(epoch_saver_trigger)
        self.set_iteration_saver_trigger(iteration_saver_trigger)

    @property
    def model(self):
        return self._model

    @property
    def iterations(self):
        return self._iterations

    @property
    def epochs(self):
        return self._epochs

    @property
    def criterion(self):
        return self._criterion

    @property
    def hooks(self):
        return self._hooks

    def set_batch_input_fn(self, fn):
        if fn is None:
            self._batch_input_fn = lambda x: x
        else:
            assert(callable(fn))
            self._batch_input_fn = fn

    def set_batch_target_fn(self, fn):
        if fn is None:
            self._batch_target_fn = lambda x: x
        else:
            assert(callable(fn))
            self._batch_target_fn = fn

    def set_early_stop_trigger(self, trigger):
        if trigger is None:
            self._early_stop_trigger = lambda: False
        else:
            assert(callable(trigger))
            self._early_stop_trigger = trigger

    def set_epoch_saver_trigger(self, trigger):
        if trigger is None:
            self._epoch_saver_trigger = lambda x: False
        else:
            assert(callable(trigger))
            self._epoch_saver_trigger = trigger

    def set_iteration_saver_trigger(self, trigger):
        if trigger is None:
            self._iteration_saver_trigger = lambda x: False
        else:
            assert(callable(trigger))
            self._iteration_saver_trigger = trigger

    def add_hook(self, when, func):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        if func is not None:
            self._hooks[when].append(func)

    def __call_hooks(self, when, **kwargs):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        for hook in self._hooks[when]:
            hook(trainer=self, **kwargs)

    def add_evaluator(self, evaluator):
        def run_eval(**kwargs):
            evaluator.run()

        if evaluator is not None:
            self.add_hook('on_end_epoch', run_eval)

    def run(self):
        saved = False
        while not self._early_stop_trigger():
            self._epochs += 1
            self.__call_hooks('on_start_epoch', epoch=self._epochs)
            for it, data in enumerate(tqdm(self._dataset), 1):
                self._iterations += 1
                batch_input = self._batch_input_fn(data)
                batch_target = self._batch_target_fn(data)
                loss_and_output = [None, None]
                self.__call_hooks('on_start_batch',
                                  epoch=self._epochs,
                                  iteration=self._iterations,
                                  batch_input=batch_input,
                                  batch_target=batch_target)

                def closure():
                    self._model.train()
                    batch_output = self._model(batch_input)
                    batch_loss = self._criterion(batch_output, batch_target)
                    batch_loss.backward()
                    loss_and_output[0] = batch_loss
                    loss_and_output[1] = batch_output
                    return batch_loss

                self._optimizer.zero_grad()
                self._optimizer.step(closure)
                self.__call_hooks('on_end_batch',
                                  epoch=self._epochs,
                                  iteration=self._iterations,
                                  batch_input=batch_input,
                                  batch_target=batch_target,
                                  batch_loss=loss_and_output[0],
                                  batch_output=loss_and_output[1])

                # After all hooks have been executed, trigger iteration saver
                if self._iteration_saver_trigger(self):
                    saved = True

            self._iterations += it
            self.__call_hooks('on_end_epoch', epoch=self._epochs)

            # After all hooks have been executed, trigger epoch saver
            if self._epoch_saver_trigger(self):
                saved = True
