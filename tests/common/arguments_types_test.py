import re
from argparse import ArgumentTypeError

import pytest

from laia.common.arguments_types import TupleList, str2loglevel, str2num


@pytest.mark.parametrize(
    ["v", "t", "vmin", "vmax", "open", "expected"],
    [
        ("0", int, 1, 2, False, "Value 0 must be in the [1, 2] interval"),
        ("1.5", float, 1, 1.5, True, "Value 1.5 must be in the (1.0, 1.5) interval"),
        ("1.5", float, 1.5, 2, True, "Value 1.5 must be in the (1.5, 2.0) interval"),
    ],
)
def test_str2num_raises(v, t, vmin, vmax, open, expected):
    with pytest.raises(ArgumentTypeError, match=re.escape(expected)):
        str2num(v, t, vmin=vmin, vmax=vmax, open=open)


@pytest.mark.parametrize(
    ["v", "t", "vmin", "vmax", "open", "expected"],
    [
        ("0", float, None, None, True, 0.0),
        ("1.5", float, 1.0, 2.0, False, 1.5),
        ("1.5", float, 1.5, 1.5, False, 1.5),
    ],
)
def test_str2num(v, t, vmin, vmax, open, expected):
    out = str2num(v, t, vmin=vmin, vmax=vmax, open=open)
    assert out == expected


def test_str2loglevel_raises():
    with pytest.raises(
        ArgumentTypeError, match=r'Valid logging levels are \[.+\], given "foo"'
    ):
        str2loglevel("foo")


@pytest.mark.parametrize(
    ["v", "expected"], [("info", 20), ("CRITICAL", 50), ("DeBuG", 10)]
)
def test_str2loglevel(v, expected):
    assert str2loglevel(v) == expected


@pytest.mark.parametrize(
    ["v", "msg"],
    [
        ("{}", "{} is neither a tuple nor <class 'int'>"),
        ("[1, 2.0]", "An element of [1, 2.0] is not a <class 'int'>"),
        (
            "[1, 2, 3]",
            "The given input [1, 2, 3] does not match the given dimensions 2",
        ),
    ],
)
def test_tuplelist_raises(v, msg):
    x = TupleList(int, dimensions=2)
    with pytest.raises(ArgumentTypeError, match=re.escape(msg)):
        x(v)


@pytest.mark.parametrize(
    ["v", "expected"],
    [
        ("1.", (1.0, 1.0, 1.0)),
        ("1.0, 2.0, 3.0", (1.0, 2.0, 3.0)),
        ("1.0,2.,3.", (1.0, 2.0, 3.0)),
        ("(1.0, 2.0, 3.0)", (1.0, 2.0, 3.0)),
        ("((1.0, 2.0, 3.0))", (1.0, 2.0, 3.0)),
        ("[1.0, 2.0, 3.0]", (1.0, 2.0, 3.0)),
    ],
)
def test_tuplelist(v, expected):
    x = TupleList(float, dimensions=3)
    assert x(v) == expected
