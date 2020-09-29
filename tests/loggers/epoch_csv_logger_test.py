import pandas as pd
import pytest
import pytorch_lightning as pl

from laia.dummies.dummy_module import DummyModule
from laia.loggers.epoch_csv_logger import EpochCSVLogger, EpochCSVWriter


@pytest.mark.parametrize(
    ["metrics", "expected"],
    [
        ([], []),
        ([{}], []),
        ([{"step": 0}], [{"epoch": 0}]),
        ([{"foo": 1, "epoch": 2}], [{"foo": 1, "epoch": 2}]),
        ([{"foo": 1, "epoch": 2}, {"foo": 1, "epoch": 2}], [{"foo": 1, "epoch": 2}]),
        (
            [{"foo": 1, "epoch": 2, "step": 0}, {"bar": 1, "epoch": 2}],
            [{"foo": 1, "bar": 1, "epoch": 2}],
        ),
        (
            [{"foo": 1, "epoch": 1}, {"bar": 1, "epoch": 2}],
            [{"foo": 1, "epoch": 1}, {"bar": 1, "epoch": 2}],
        ),
        ([{"foo": 1, "epoch": 1}, {"foo": 2, "epoch": 1}], [{"foo": 2, "epoch": 1}]),
        ([{"foo": 2, "epoch": 1}, {"foo": 1, "epoch": 1}], [{"foo": 1, "epoch": 1}]),
        (
            [
                {"foo": 2, "epoch": 2},
                {"foo": 1, "epoch": 1},
                {"bar": 2, "step": 1},
                {"bar": 1, "epoch": 2},
            ],
            [{"foo": 1, "bar": 2, "epoch": 1}, {"bar": 1, "foo": 2, "epoch": 2}],
        ),
    ],
)
def test_group_by_epoch(metrics, expected):
    assert EpochCSVWriter.group_by_epoch(metrics) == expected


@pytest.mark.parametrize(
    ["files", "expected"],
    [
        ([], -1),
        (["metrics.csv"], 0),
        (["metrics-v10.csv"], 11),
        (["metrics.csv", "metrics-v0.csv"], 1),
        (["metrics-v-10.csv"], -1),
        (["metrics.csv", "foo", "bar", "metrics-v3.csv"], 4),
    ],
)
def test_get_next_version(tmpdir, files, expected):
    for f in files:
        (tmpdir / f).write(None)
    assert EpochCSVLogger.get_next_version(tmpdir) == expected


@pytest.mark.parametrize("num_processes", (1, 2))
def test_epoch_csv_logger(tmpdir, num_processes):
    pl.seed_everything(0)

    # run twice
    for csv_filename in ("metrics.csv", "metrics-v0.csv"):
        trainer = pl.Trainer(
            default_root_dir=tmpdir,
            checkpoint_callback=False,
            max_epochs=3,
            weights_summary=None,
            limit_train_batches=10,
            progress_bar_refresh_rate=0,
            distributed_backend="ddp_cpu" if num_processes > 1 else None,
            num_processes=num_processes,
            logger=EpochCSVLogger(tmpdir),
        )
        trainer.fit(DummyModule(batch_size=2))

        csv = pd.read_csv(tmpdir / csv_filename)
        # check epoch values
        assert list(csv["epoch"].values) == [
            float(i) for i in range(trainer.max_epochs)
        ]
        # check test variable "bar" values
        assert list(csv["bar"].values) == [
            float(i)
            for i in range(
                trainer.limit_train_batches - 1,
                (trainer.limit_train_batches * trainer.max_epochs),
                trainer.limit_train_batches,
            )
        ]
        # check losses are floats
        assert all(isinstance(v, float) for v in csv["tr_loss"].values)
