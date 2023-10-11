import pytorch_lightning as pl


class DummyTrainer(pl.Trainer):
    def __init__(self, **kwargs):
        defaults = {
            "checkpoint_callback": False,
            "logger": True,
            "weights_summary": None,
            "max_epochs": 1,
            "limit_train_batches": 10,
            "limit_val_batches": 10,
            "limit_test_batches": 10,
            "progress_bar_refresh_rate": 0,
            "deterministic": True,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
