from __future__ import absolute_import

import unittest

from laia.engine.engine import Engine, EPOCH_START, EPOCH_END, ITER_START, ITER_END
from laia.hooks import action


class DummyModel(object):
    def __init__(self):
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        return x


class EngineTest(unittest.TestCase):
    def test_simple(self):
        model = DummyModel()
        engine = Engine(model=model, data_loader=[1, 2, 3, 4])
        engine.run()
        self.assertEqual(4, model.counter)

    def test_progress_bar(self):
        model = DummyModel()
        engine = Engine(model=model, data_loader=[1, 2, 3, 4], progress_bar=True)
        engine.run()
        self.assertEqual(4, model.counter)

    def test_progress_bar_description(self):
        model = DummyModel()
        engine = Engine(
            model=model, data_loader=[1, 2, 3, 4], progress_bar="Description"
        )
        engine.run()
        self.assertEqual(4, model.counter)

    def test_hooks(self):
        counters = [0, 0, 0, 0]

        @action
        def on_iter_start():
            counters[0] += 1

        @action
        def on_epoch_start():
            counters[1] += 1

        @action
        def on_iter_end():
            counters[2] += 1

        @action
        def on_epoch_end():
            counters[3] += 1

        engine = Engine(model=lambda x: x, data_loader=[1, 2])
        engine.add_hook(ITER_START, on_iter_start)
        engine.add_hook(EPOCH_START, on_epoch_start)
        engine.add_hook(ITER_END, on_iter_end)
        engine.add_hook(EPOCH_END, on_epoch_end)
        engine.run()
        engine.run()
        # Check the number of calls to each function
        self.assertEqual([4, 2, 4, 2], counters)

    def test_reset(self):
        engine = Engine(model=lambda x: x, data_loader=[1, 2, 3])
        self.assertEqual(0, engine.epochs())
        self.assertEqual(0, engine.iterations())
        engine.run()
        self.assertEqual(1, engine.epochs())
        self.assertEqual(3, engine.iterations())
        engine.run()
        self.assertEqual(2, engine.epochs())
        self.assertEqual(6, engine.iterations())
        engine.reset()
        self.assertEqual(0, engine.epochs())
        self.assertEqual(0, engine.iterations())


if __name__ == "__main__":
    unittest.main()
