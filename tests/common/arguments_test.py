import argparse
import sys

import laia.common.arguments as args
import laia.common.logging as log


def test_get_key():
    assert args.get_key({"learning_rate": 1.0}, "learning_rate") == 1.0
    assert args.get_key({}, "learning_rate") == 5e-4
    assert not args.get_key({"foo": "bar"}, "nesterov")
    assert args.get_key({"foo": "bar", "nesterov": True}, "nesterov")


def test_valid_arguments():
    parser = argparse.ArgumentParser()
    for names, kwargs in args.default_args.values():
        parser.add_argument(*names, **kwargs)
    parser.parse_args([])


def test_get_parser():
    parser1 = args._get_parser()
    parser2 = args._get_parser()
    assert parser1 is parser2


def test_add_defaults():
    parser = args.add_defaults("nesterov", "momentum", momentum=3)
    parsed = parser.parse_args([])
    assert not parsed.nesterov
    assert parsed.momentum == 3.0
    args.parser = None  # reset global


def test_args(caplog, monkeypatch):
    args.add_defaults()
    monkeypatch.setattr(
        "sys.argv", []
    )  # pytest automatically sets argv which is used later in parse_args()
    assert log.get_logger().level == 0
    args.args()
    assert log.get_logger().level == log.INFO
    assert len(caplog.messages) == 1
    assert eval(caplog.messages[0]) == {
        "logging_also_to_stderr": 40,
        "logging_file": None,
        "logging_level": 20,
        "logging_overwrite": False,
        "print_args": True,
    }
    log.clear()  # reset logging
