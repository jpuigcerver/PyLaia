import pandas as pd
import pytest
import pytorch_lightning as pl

from laia.dummies import DummyModule, DummyTrainer
from laia.loggers.epoch_csv_logger import EpochCSVLogger, EpochCSVWriter


@pytest.mark.parametrize(
    ["dicts", "key", "expected"],
    [
        ([], None, []),
        ([{}], None, []),
        ([{"foo": 1}], None, []),
        ([{"foo": 1}], "foo", [{"foo": 1}]),
        ([{"foo": 1, "bar": 2}], "foo", [{"foo": 1, "bar": 2}]),
        ([{"foo": 1}, {"foo": 2}], "foo", [{"foo": 1}, {"foo": 2}]),
        ([{"foo": 1}, {"foo": 1}], "foo", [{"foo": 1}]),
        ([{"foo": 1, "bar": 2}, {"foo": 1, "bar": 3}], "foo", [{"foo": 1, "bar": 3}]),
    ],
)
def test_merge_by(dicts, key, expected):
    assert EpochCSVWriter.merge_by(dicts, key) == expected


@pytest.mark.parametrize(
    ["metrics", "expected"],
    [
        ([], []),
        ([{}], []),
        ([{"step": 0}], [{"epoch": 0}]),
        ([{"step": 1, "epoch": 2}], [{"epoch": 2}]),
        ([{"step": 1, "epoch": 2}, {"step": 1, "epoch": 2}], [{"epoch": 2}]),
        (
            [{"foo": 1, "epoch": 2, "step": 0}, {"bar": 1, "epoch": 2, "step": 1}],
            [{"foo": 1, "bar": 1, "epoch": 2}],
        ),
        (
            [{"foo": 1, "epoch": 1, "step": 0}, {"bar": 1, "epoch": 2, "step": 1}],
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
        trainer = DummyTrainer(
            default_root_dir=tmpdir,
            max_epochs=3,
            distributed_backend="ddp_cpu" if num_processes > 1 else None,
            num_processes=num_processes,
            logger=EpochCSVLogger(tmpdir),
        )
        trainer.fit(DummyModule(batch_size=2))

        csv = pd.read_csv(tmpdir / csv_filename)
        # check epoch values
        assert (
            list(csv["epoch"].values)
            == list(csv["foo"].values)
            == list(range(trainer.max_epochs))
        )
        # check test variable "bar" values
        assert list(csv["bar"].values) == list(
            range(
                trainer.limit_train_batches - 1,
                (trainer.limit_train_batches * trainer.max_epochs),
                trainer.limit_train_batches,
            )
        )
        # check losses are floats
        assert all(isinstance(v, float) for v in csv["tr_loss"].values)
