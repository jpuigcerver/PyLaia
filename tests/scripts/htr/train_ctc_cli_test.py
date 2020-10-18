import argparse
import shutil
import subprocess
from unittest import mock

import laia.common.logging as log
from laia.scripts.htr.train_ctc import get_args


def test_get_args(tmpdir):
    syms = tmpdir / "syms"
    syms.write(None)
    img_dirs = [str(tmpdir / p) for p in ("tr", "va", "te")]
    txt_table = tmpdir / "gt"
    txt_table.write(None)

    cmd_args = ["ignored", str(syms), *img_dirs, str(txt_table), str(txt_table)]
    with mock.patch("sys.argv", new=cmd_args):
        args = get_args()
    log.clear()

    assert isinstance(args, argparse.Namespace)
    assert isinstance(args.lightning, argparse.Namespace)
    assert len(args.img_dirs) == 3
    assert args.tr_txt_table.name == args.va_txt_table.name


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-train-ctc"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout.decode().startswith("usage: pylaia-htr-train-ctc")
