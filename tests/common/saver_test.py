from pathlib import Path

import pytest
import torch

from laia.common.saver import BasicSaver, ObjectSaver


def test_basic_saver(tmpdir):
    saver = BasicSaver()
    saver.save(None, tmpdir / "test.pth")
    # with extra non-existing dir
    saver.save(None, tmpdir / "extra" / "test.pth")
    # again to test exists_ok
    saver.save(None, tmpdir / "extra" / "test.pth")


class Foo:
    pass


@pytest.mark.parametrize(
    ["cls", "args", "kwargs", "expected"],
    [
        (
            Foo,
            [],
            {},
            {"module": __name__, "name": "Foo", "args": tuple(), "kwargs": {}},
        ),
        (Foo, [1], {}, {"module": __name__, "name": "Foo", "args": (1,), "kwargs": {}}),
        (
            Foo,
            [1, 2, 3],
            {},
            {"module": __name__, "name": "Foo", "args": (1, 2, 3), "kwargs": {}},
        ),
        (
            Foo,
            [],
            {"foo": 1, "bar": 2},
            {
                "module": __name__,
                "name": "Foo",
                "args": tuple(),
                "kwargs": {"foo": 1, "bar": 2},
            },
        ),
    ],
)
def test_object_saver(tmpdir, cls, args, kwargs, expected):
    f = Path(tmpdir) / "test.pth"
    saver = ObjectSaver(f)
    saver.save(cls, *args, **kwargs)
    assert torch.load(f) == expected
