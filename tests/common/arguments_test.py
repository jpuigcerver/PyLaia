from re import escape

import pytest
import pytorch_lightning as pl
import torch

from laia.common.arguments import CreateCRNNArgs, TrainerArgs


def test_trainer_args():
    args = TrainerArgs()
    assert not hasattr(args, "callbacks")
    # instantiate to check if its valid
    pl.Trainer(**vars(args))


def test_trainer_args_postinit(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "1.5.0")
    with pytest.raises(ValueError, match=r"AMP requires torch>=1\.7\.0"):
        TrainerArgs(precision=15)


def test_createcrnn_args_postinit():
    with pytest.raises(ValueError, match="Wrong cnn layer dimensions"):
        CreateCRNNArgs(cnn_num_features=[1])
    with pytest.raises(ValueError, match="Could not find all cnn activations"):
        CreateCRNNArgs(cnn_activation=["ReLU"] * 3 + ["RELU"])
    with pytest.raises(ValueError, match="Could not find RNN type"):
        CreateCRNNArgs(rnn_type="LSTN")


@pytest.mark.parametrize(
    ["v", "msg"],
    [
        ([{}], "{} (<class 'dict'>) is neither a tuple nor an int"),
        ([1, 2.0], "2.0 (<class 'float'>) is neither a tuple nor an int"),
        (
            [(1, 2, 3)],
            "The given input [(1, 2, 3)] does not match the given dimensions 2",
        ),
    ],
)
def test_parse_parameter_raises(v, msg):
    with pytest.raises(ValueError, match=escape(msg)):
        CreateCRNNArgs.parse_parameter(v)


@pytest.mark.parametrize(
    ["v", "expected"],
    [
        ([1], [[1] * 3]),
        ([1, 2, 3], [[1] * 3, [2] * 3, [3] * 3]),
        ([(1, 2, 3)], [[1, 2, 3]]),
        ([[1, 2, 3]], [[1, 2, 3]]),
    ],
)
def test_parse_parameter(v, expected):
    assert CreateCRNNArgs.parse_parameter(v, dim=3) == expected
