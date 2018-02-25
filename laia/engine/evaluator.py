from __future__ import absolute_import

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


class Evaluator(object):
    def __init__(self, model, dataset, batch_input_fn=None,
                 batch_target_fn=None):
        self._model = model
        self._dataset = dataset
        self._epochs = 0
        self._batch_input_fn = None
        self._batch_target_fn = None
        self._hooks = {
            'on_start_batch': [],
            'on_start_epoch': [],
            'on_end_batch': [],
            'on_end_epoch': [],
        }

        # Default functions
        self.set_batch_input_fn(batch_input_fn)
        self.set_batch_target_fn(batch_target_fn)

    @property
    def model(self):
        return self._model

    @property
    def hooks(self):
        return self._hooks

    @property
    def epochs(self):
        return self._epochs

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

    def add_hook(self, when, func):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        self._hooks[when].append(func)

    def __call_hooks(self, when, **kwargs):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        for hook in self._hooks[when]:
            hook(trainer=self, **kwargs)

    def run(self):
        self._epochs += 1
        self.__call_hooks('on_start_epoch')
        for it, data in enumerate(tqdm(self._dataset), 1):
            batch_input = self._batch_input_fn(data)
            batch_target = self._batch_target_fn(data)
            self.__call_hooks('on_start_batch',
                              iteration=it,
                              batch=data,
                              batch_input=batch_input,
                              batch_target=batch_target)

            # Put model in eval mode
            if hasattr(self._model, 'eval'):
                self._model.eval()
            batch_output = self._model(batch_input)

            self.__call_hooks('on_end_batch',
                              iteration=it,
                              batch=data,
                              batch_input=batch_input,
                              batch_target=batch_target,
                              batch_output=batch_output)
        self.__call_hooks('on_end_epoch')
