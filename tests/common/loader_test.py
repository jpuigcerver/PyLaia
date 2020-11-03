import os
import shutil
from collections import OrderedDict
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from laia.common.loader import ModelLoader, ObjectLoader
from laia.dummies import DummyEngine, DummyMNIST, DummyTrainer


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
def test_object_loader_raises(tmpdir, input, exception):
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
def test_model_loader_get_model_state_dict_raises(tmpdir, input, exception):
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
    assert ModelLoader.choose_by(f"{tmpdir}/{input}") is None


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


def test_model_loader_find_best(tmpdir):
    # empty directory
    assert ModelLoader.find_best(tmpdir, "test") is None

    # with no-monitor ckpts
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=tmpdir, save_top_k=-1)],
        checkpoint_callback=True,
        max_epochs=3,
    )
    trainer.fit(DummyEngine(), datamodule=DummyMNIST())
    assert ModelLoader.find_best(tmpdir, "test") is None

    # with monitor ckpts
    monitor = "bar"
    mc = pl.callbacks.ModelCheckpoint(
        dirpath=tmpdir, save_top_k=-1, monitor=monitor, mode="max"
    )
    trainer = DummyTrainer(
        default_root_dir=tmpdir, callbacks=[mc], checkpoint_callback=True, max_epochs=3
    )
    trainer.fit(DummyEngine(), datamodule=DummyMNIST())
    assert (
        ModelLoader.find_best(tmpdir, monitor, mode="max")
        == tmpdir / "epoch=2-v0.ckpt"
        == mc.best_model_path
    )
    assert (
        ModelLoader.find_best(tmpdir, monitor, mode="min") == tmpdir / "epoch=0-v0.ckpt"
    )


def test_model_loader_prepare_checkpoint(tmpdir, caplog):
    # create some checkpoints
    monitor = "bar"
    exp_dirpath = tmpdir / "experiment"
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=exp_dirpath, save_top_k=-1, monitor=monitor, mode="max"
            )
        ],
        checkpoint_callback=True,
        max_epochs=2,
    )
    trainer.fit(DummyEngine(), datamodule=DummyMNIST())

    expected = exp_dirpath / "epoch=0.ckpt"
    # nothing
    assert ModelLoader.prepare_checkpoint("", exp_dirpath, monitor) == expected
    # direct path
    assert ModelLoader.prepare_checkpoint(expected, exp_dirpath, monitor) == expected
    # direct path outside of exp_dirpath
    shutil.copy(expected, "/tmp")
    assert (
        ModelLoader.prepare_checkpoint("/tmp/epoch=0.ckpt", exp_dirpath, monitor)
        == "/tmp/epoch=0.ckpt"
    )
    # filename
    assert (
        ModelLoader.prepare_checkpoint("epoch=0.ckpt", exp_dirpath, monitor) == expected
    )
    # globbed filename
    assert (
        ModelLoader.prepare_checkpoint("epoch=?.ckpt", exp_dirpath, monitor)
        == exp_dirpath / "epoch=1.ckpt"
    )
    # failures
    with pytest.raises(SystemExit):
        ModelLoader.prepare_checkpoint("", tmpdir, monitor)
        assert caplog.messages[-1].startswith("Could not find a valid checkpoint in")
    with pytest.raises(SystemExit):
        ModelLoader.prepare_checkpoint("?", exp_dirpath, monitor)
        assert caplog.messages[-1].startswith("Could not find the checkpoint")
