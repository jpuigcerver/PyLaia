from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.trainer

class Trainer(torch.utils.trainer.Trainer):
    def __init__(self, model, criterion, optimizer, dataset,
                 batch_input_fn, batch_target_fn):
        super(Trainer, self).__init__(model=model, criterion=criterion,
                                      optimizer=optimizer, dataset=dataset)
        if batch_input_fn is None:
            self._batch_input_fn = lambda x: x
        else:
            self._batch_input_fn = batch_input_fn

        if batch_target_fn is None:
            self._batch_target_fn = batch_target_fn
        else:
            self._batch_target_fn = batch_target_fn

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input = self._batch_input_fn(data)
            batch_target = self._batch_target_fn(data)
            self.call_plugins('batch', i, batch_input, batch_target)
            plugin_data = [None, None]

            def closure():
                batch_output = self.model(Variable(batch_input))
                loss = self.criterion(batch_output, batch_target)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i
