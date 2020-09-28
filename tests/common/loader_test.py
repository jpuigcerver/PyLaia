import os
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from laia.common.loader import ModelLoader, ObjectLoader


class Foo:
    def __init__(self, arg, *args, kwarg=None, **kwargs):
        self.arg = arg
        self.args = args
        self.kwarg = kwarg
        self.kwargs = kwargs

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self.arg == other.arg
            and self.args == other.args
            and self.kwarg == other.kwarg
            and self.kwargs == other.kwargs
        )


def test_object_loader_no_object(tmpdir):
    f = Path(tmpdir) / "test.pth"
    loader = ObjectLoader(f)
    assert loader.load() is None


@pytest.mark.parametrize(
    ["input", "exception"],
    [
        ({}, KeyError),
        ({"module": __name__}, KeyError),
        ({"module": "foo"}, ModuleNotFoundError),
        ({"module": __name__, "name": "foo"}, AttributeError),
        ({"module": __name__, "name": "Foo"}, TypeError),
        ({"module": __name__, "name": "Foo", "kwargs": {"bar": 1}}, TypeError),
    ],
)
def test_object_loader_fails(tmpdir, input, exception):
    f = Path(tmpdir) / "test.pth"
    torch.save(input, f)
    loader = ObjectLoader(f)
    with pytest.raises(exception):
        loader.load()


@pytest.mark.parametrize(
    ["input", "expected"],
    [
        (
            {"module": __name__, "name": "Foo", "args": [1], "kwargs": {"kwarg": 2}},
            Foo(1, kwarg=2),
        ),
        (
            {"module": __name__, "name": "Foo", "args": [1, 2]},
            Foo(1, 2, kwarg=None),
        ),
        (
            {"module": __name__, "name": "Foo", "args": [1, 2], "kwargs": {"kwarg": 3}},
            Foo(1, 2, kwarg=3),
        ),
        (
            {
                "module": __name__,
                "name": "Foo",
                "args": [1, 2, 3],
                "kwargs": {"kwarg": 4, "bar": 5},
            },
            Foo(1, 2, 3, bar=5, kwarg=4),
        ),
    ],
)
def test_object_loader(tmpdir, input, expected):
    f = Path(tmpdir) / "test.pth"
    torch.save(input, f)
    loader = ObjectLoader(f)
    assert loader.load() == expected


@pytest.mark.parametrize(
    ["input", "exception"],
    [
        ({"tr_engine": {"foo": 0, "iterations": 1, "model": "old_ckpt"}}, KeyError),
        (
            {
                "pytorch-lightning_version": -1,
            },
            KeyError,
        ),
        (
            {
                "pytorch-lightning_version": -1,
                "epoch": 0,
                "global_step": 1,
                "state_dict": {"foo.bar": torch.tensor(1)},
            },
            AssertionError,
        ),
    ],
)
def test_model_loader_get_model_state_dict_fails(tmpdir, input, exception):
    f = Path(tmpdir) / "checkpoint.ckpt"
    torch.save(input, f)
    loader = ModelLoader(tmpdir)
    with pytest.raises(exception):
        loader.get_model_state_dict(f)


@pytest.mark.parametrize(
    ["input", "expected"],
    [
        ({}, {}),
        (
            {"tr_engine": {"epochs": 0, "iterations": 1, "model": "old_ckpt"}},
            "old_ckpt",
        ),
        (
            {
                "pytorch-lightning_version": -1,
                "epoch": 0,
                "global_step": 1,
                "state_dict": {"model.foo": torch.tensor(1)},
            },
            OrderedDict(foo=torch.tensor(1)),
        ),
    ],
)
def test_model_loader_get_model_state_dict(tmpdir, input, expected):
    f = Path(tmpdir) / "checkpoint.ckpt"
    torch.save(input, f)
    loader = ModelLoader(tmpdir)
    state_dict = loader.get_model_state_dict(f)
    assert state_dict == expected


@pytest.mark.parametrize("input", ["", "foo"])
def test_model_loader_choose_by_empty(tmpdir, input):
    assert ModelLoader.choose_by(f"{tmpdir}/{input}") == None


@pytest.mark.parametrize(
    ["input", "expected"],
    [
        ("*", "test-14.ckpt"),
        ("test-12.ckpt", "test-12.ckpt"),
        ("test-?.ckpt", "test-9.ckpt"),
        ("test-*.ckpt", "test-14.ckpt"),
    ],
)
def test_model_loader_choose_by(tmpdir, input, expected):
    n_files = 15
    for i in range(n_files):
        f = Path(tmpdir) / f"test-{i}.ckpt"
        torch.save(None, f)
    assert len(os.listdir(tmpdir)) == n_files
    assert ModelLoader.choose_by(f"{tmpdir}/{input}") == f"{tmpdir}/{expected}"
