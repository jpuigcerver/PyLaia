import pytest

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
