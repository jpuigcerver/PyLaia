import argparse
import shutil
import subprocess
from unittest import mock

import laia.common.logging as log
from laia.scripts.htr.netout import get_args


def test_get_args(tmpdir):
    img_list = tmpdir / "list"
    img_list.write(None)
    ckpt = tmpdir / "model.ckpt"
    ckpt.write(None)
    img_dirs = tmpdir / "imgs"

    cmd_args = ["ignored", str(img_list), str(ckpt), str(img_dirs)]
    with mock.patch("sys.argv", new=cmd_args):
        args = get_args()
    log.clear()

    assert isinstance(args, argparse.Namespace)
    assert isinstance(args.lightning, argparse.Namespace)
    assert len(args.img_dirs) == 1


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-netout"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout.decode().startswith("usage: pylaia-htr-netout")
