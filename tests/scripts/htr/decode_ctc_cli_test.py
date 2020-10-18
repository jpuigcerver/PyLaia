import argparse
import shutil
import subprocess
from unittest import mock

import laia.common.logging as log
from laia.scripts.htr.decode_ctc import get_args


def test_get_args(tmpdir):
    syms = tmpdir / "syms"
    syms.write(None)
    img_list = tmpdir / "list"
    img_list.write(None)
    ckpt = tmpdir / "model.ckpt"
    ckpt.write(None)

    cmd_args = [
        "ignored",
        str(syms),
        str(img_list),
        str(ckpt),
        "--use_symbols=f",
        "--convert_spaces",
        "--color_mode",
        "RGBA",
    ]
    with mock.patch("sys.argv", new=cmd_args):
        args = get_args()
    log.clear()

    assert isinstance(args, argparse.Namespace)
    assert isinstance(args.lightning, argparse.Namespace)
    assert len(args.img_dirs) == 0
    assert not args.use_symbols
    assert args.convert_spaces
    assert args.color_mode == "RGBA"


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-decode-ctc"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout.decode().startswith("usage: pylaia-htr-decode-ctc")
