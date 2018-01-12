from __future__ import print_function

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

class Evaluator(object):
    def __init__(self, model, dataset, batch_input_fn=None,
                 batch_target_fn=None):
        self._model = model
        self._dataset = dataset
        self._batch_input_fn = batch_input_fn
        self._batch_target_fn = batch_target_fn
        self._hooks = {
            'on_start_batch': [],
            'on_start_epoch': [],
            'on_end_batch': [],
            'on_end_epoch': [],
        }

        # Default functions
        if batch_input_fn is None:
            self._batch_input_fn = lambda x: x
        if batch_target_fn is None:
            self._batch_target_fn = lambda x: x

    @property
    def model(self):
        return self._model

    @property
    def hooks(self):
        return self._hooks

    def add_hook(self, when, func):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        self._hooks[when].append(func)

    def __call_hooks(self, when, **kwargs):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        for hook in self._hooks[when]:
            hook(trainer=self, **kwargs)

    def run(self):
        self.__call_hooks('on_start_epoch')
        for it, data in enumerate(tqdm(self._dataset), 1):
            batch_input = self._batch_input_fn(data)
            batch_target = self._batch_target_fn(data)
            self.__call_hooks('on_start_batch',
                              iteration=it,
                              batch_input=batch_input,
                              batch_target=batch_target)

            self._model.eval()
            batch_output = self._model(batch_input)

            self.__call_hooks('on_end_batch',
                              iteration=it,
                              batch_input=batch_input,
                              batch_target=batch_target,
                              batch_output=batch_output)
        self.__call_hooks('on_end_epoch')
