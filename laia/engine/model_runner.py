from __future__ import absolute_import

from torch._six import string_classes

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class ModelRunner(object):
    ON_BATCH_START = 'ON_BATCH_START'
    ON_BATCH_END   = 'ON_BATCH_END'
    ON_EPOCH_START = 'ON_EPOCH_START'
    ON_EPOCH_END   = 'ON_EPOCH_END'


    def __init__(self, model, data_loader, batch_input_fn=None, progress_bar=None):
        self._model = model
        self._data_loader = data_loader
        self._batch_input_fn = None
        self._progress_bar = progress_bar

        self._epochs = 0
        self._iterations = 0
        self.set_batch_input_fn(batch_input_fn)

        self._hooks = {
            self.ON_BATCH_START: [],
            self.ON_EPOCH_START: [],
            self.ON_BATCH_END: [],
            self.ON_EPOCH_END: []
        }

    @property
    def model(self):
        return self._model

    @property
    def hooks(self):
        return self._hooks

    @property
    def epochs(self):
        return self._epochs

    @property
    def iterations(self):
        return self._iterations

    def set_batch_input_fn(self, fn):
        if fn is None:
            self._batch_input_fn = lambda x: x
        else:
            assert(callable(fn))
            self._batch_input_fn = fn

    def add_hook(self, when, func):
        assert when in self._hooks, (
            '{!r} is not a valid hook event'.format(when))
        self._hooks[when].append(func)

    def _call_hooks(self, when, **kwargs):
        for hook in self._hooks[when]:
            hook(caller=self, **kwargs)

    def reset(self):
        self._epochs = 0
        self._iterations = 0

    def _run_iteration(self, it, batch):
        self._iterations += 1

        batch_input = self._batch_input_fn(batch)

        self._call_hooks(self.ON_BATCH_START,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input)

        # Put model in evaluation mode
        if hasattr(self._model, 'eval'):
            self._model.eval()

        batch_output = self._model(batch_input)

        self._call_hooks(self.ON_BATCH_END,
                         batch=batch,
                         batch_num=it,
                         epoch=self._epochs,
                         iteration=self._iterations,
                         batch_input=batch_input,
                         batch_output=batch_output)

    def _run_epoch(self):
        self._epochs += 1
        self._call_hooks(self.ON_EPOCH_START, epoch=self._epochs)

        if self._progress_bar and tqdm:
            if isinstance(self._progress_bar, string_classes):
                batch_iterator = tqdm(self._data_loader,
                                      desc=self._progress_bar)
            else:
                batch_iterator = tqdm(self._data_loader)
        else:
            batch_iterator = self._data_loader


        for it, batch in enumerate(batch_iterator, 1):
            self._run_iteration(it, batch)

        self._call_hooks(self.ON_EPOCH_END, epoch=self._epochs)

    def run(self):
        self._run_epoch()
