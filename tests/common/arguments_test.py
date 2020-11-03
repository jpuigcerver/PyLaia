import argparse
from ast import literal_eval

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


def test_add_defaults():
    parser = args.LaiaParser().add_defaults("nesterov", "momentum", momentum=3)
    parsed = parser.parse_args([], should_log=False)
    assert not parsed.nesterov
    assert parsed.momentum == 3.0


def test_args(caplog, monkeypatch):
    parser = args.LaiaParser().add_defaults(
        "logging_also_to_stderr",
        "logging_file",
        "logging_level",
        "logging_overwrite",
        "print_args",
    )
    assert log.get_logger().level == 0
    parser.parse_args([])
    assert log.get_logger().level == log.INFO
    assert len(caplog.messages) == 1
    assert literal_eval(caplog.messages[0]) == {
        "logging_also_to_stderr": 40,
        "logging_file": None,
        "logging_level": 20,
        "logging_overwrite": False,
        "print_args": True,
    }
    log.clear()  # reset logging
