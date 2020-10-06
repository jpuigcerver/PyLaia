import time

import pytorch_lightning as pl

import laia.common.logging as log
from laia.callbacks import TrainingTimer
from laia.dummies import DummyModule, DummyTrainer

i = 1


def test(monkeypatch, caplog):
    def fake_time():
        global i
        i += 1
        return i

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(TrainingTimer, "time_to_str", staticmethod(lambda x: str(x)))
    log.get_logger("laia.callbacks.training_timer").setLevel(log.INFO)

    class TestCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, *args):
            assert sum(isinstance(c, TrainingTimer) for c in trainer.callbacks) == 1
            for c in trainer.callbacks:
                if isinstance(c, TrainingTimer):
                    assert c.tr_timer.start == i
                    assert c.tr_timer.end is None

        def on_validation_epoch_start(self, trainer, pl_module):
            assert sum(isinstance(c, TrainingTimer) for c in trainer.callbacks) == 1
            for c in trainer.callbacks:
                if isinstance(c, TrainingTimer):
                    assert c.va_timer.start == i
                    assert c.va_timer.end is None

    trainer = DummyTrainer(max_epochs=2, callbacks=[TrainingTimer(), TestCallback()])
    trainer.fit(DummyModule())

    for e in range(trainer.max_epochs):
        assert caplog.messages.count(f"Epoch {e}: tr_time=2, va_time=2") == 1
