import argparse
import shutil
import subprocess
from unittest import mock

import pytest

import laia.common.logging as log
from laia.scripts.htr.create_model import get_args


@pytest.mark.parametrize(
    "cmd_args",
    [
        [],
        # model used to train IAM-htr
        # fmt: off
        [
            "--cnn_num_features", "16", "32", "48", "64", "80",
            "--cnn_kernel_size", *(["3"] * 5),
            "--cnn_stride", *(["1"] * 5),
            "--cnn_dilation", *(["1"] * 5),
            "--cnn_activation", *(["LeakyReLU"] * 5),
            "--cnn_poolsize", *(["2"] * 3), *(["0"] * 2),
            "--cnn_dropout", *(["0"] * 5),
            "--cnn_batchnorm", *(["f"] * 5),
            "--rnn_units=256",
            "--rnn_layers=5",
            "--adaptive_pooling=avgpool-16",
            "--fixed_input_height=128",
        ],
        # fmt: on
    ],
)
def test(tmpdir, cmd_args):
    syms = tmpdir / "syms"
    syms.write(None)

    # first argument would be the script name, but it is ignored
    cmd_args = ["ignored", "1", str(syms)] + cmd_args
    with mock.patch("sys.argv", new=cmd_args):
        args = get_args()
    log.clear()

    assert isinstance(args, argparse.Namespace)


def test_entry_point():
    # check that the console script entry point works
    proc = subprocess.run(
        [shutil.which("pylaia-htr-create-model"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout.decode().startswith("usage: pylaia-htr-create-model")
