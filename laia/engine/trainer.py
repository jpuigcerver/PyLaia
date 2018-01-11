from __future__ import print_function

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

class Trainer(object):
    def __init__(self, model, criterion, optimizer, dataset,
                 batch_input_fn=None, batch_target_fn=None,
                 early_stop_fn=None):
        self._model = model
        self._criterion = criterion
        self._dataset = dataset
        self._optimizer = optimizer
        self._iterations = 0
        self._epochs = 0
        self._batch_input_fn = batch_input_fn
        self._batch_target_fn = batch_target_fn
        self._early_stop_fn = early_stop_fn
        self._hooks = {
            'on_start_epoch': [],
            'on_start_batch': [],
            'on_end_epoch': [],
            'on_end_batch': [],
        }

        # Default functions
        if batch_input_fn is None:
            self._batch_input_fn = lambda x: x
        if batch_target_fn is None:
            self._batch_target_fn = lambda x: x
        if early_stop_fn is None:
            self._early_stop_fn = lambda x: False

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

    def register_hook(self, when, func):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        self._hooks[when].append(func)

    def __call_hooks(self, when, **kwargs):
        assert when in self._hooks, '"%s" is not a valid hook event' % when
        for hook in self._hooks[when]:
            hook(trainer=self, **kwargs)

    def run(self):
        while not self._early_stop_fn(self):
            self._epochs += 1
            self.__call_hooks('on_start_epoch', epoch=self._epochs)
            for it, data in enumerate(tqdm(self._dataset),
                                      self._iterations + 1):
                batch_input = self._batch_input_fn(data)
                batch_target = self._batch_target_fn(data)
                loss_and_output = [None, None]
                self.__call_hooks('on_start_batch',
                                  epoch=self._epochs,
                                  iteration=it,
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
                                  iteration=it,
                                  batch_input=batch_input,
                                  batch_target=batch_target,
                                  batch_loss=loss_and_output[0],
                                  batch_output=loss_and_output[1])

            self._iterations += it
            self.__call_hooks('on_end_epoch', epoch=self._epochs)
