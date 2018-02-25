from __future__ import absolute_import

from laia.engine.model_runner import ModelRunner

import unittest


class DummyModel(object):
    def __init__(self):
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        return x

class ModelRunnerTest(unittest.TestCase):
    def test_simple(self):
        model = DummyModel()
        runner = ModelRunner(model=model,
                             data_loader=[1, 2, 3, 4])
        runner.run()
        self.assertEqual(4, model.counter)

    def test_progress_bar(self):
        model = DummyModel()
        runner = ModelRunner(model=model,
                             data_loader=[1, 2, 3, 4],
                             progress_bar=True)
        runner.run()
        self.assertEqual(4, model.counter)

    def test_progress_bar_description(self):
        model = DummyModel()
        runner = ModelRunner(model=model,
                             data_loader=[1, 2, 3, 4],
                             progress_bar='Description')
        runner.run()
        self.assertEqual(4, model.counter)

    def test_hooks(self):
        counters = [0, 0, 0, 0]
        def on_batch_start(caller, epoch, **kwargs):
            counters[0] += 1
        def on_epoch_start(caller, epoch, **kwargs):
            counters[1] += 1
        def on_batch_end(caller, epoch, **kwargs):
            counters[2] += 1
        def on_epoch_end(caller, epoch, **kwargs):
            counters[3] += 1

        runner = ModelRunner(model=lambda x: x,
                             data_loader=[1, 2])
        runner.add_hook(ModelRunner.ON_BATCH_START, on_batch_start)
        runner.add_hook(ModelRunner.ON_EPOCH_START, on_epoch_start)
        runner.add_hook(ModelRunner.ON_BATCH_END, on_batch_end)
        runner.add_hook(ModelRunner.ON_EPOCH_END, on_epoch_end)
        runner.run()
        runner.run()
        # Check the number of calls to each function
        self.assertEqual([4, 2, 4, 2], counters)

    def test_reset(self):
        runner = ModelRunner(model=lambda x: x,
                             data_loader=[1, 2, 3])
        self.assertEqual(0, runner.epochs)
        self.assertEqual(0, runner.iterations)
        runner.run()
        self.assertEqual(1, runner.epochs)
        self.assertEqual(3, runner.iterations)
        runner.run()
        self.assertEqual(2, runner.epochs)
        self.assertEqual(6, runner.iterations)
        runner.reset()
        self.assertEqual(0, runner.epochs)
        self.assertEqual(0, runner.iterations)


if __name__ == '__main__':
    unittest.main()
