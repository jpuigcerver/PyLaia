import shutil
import subprocess
from enum import Enum

import pytest

from laia.common.arguments import DecodeArgs
from laia.scripts.htr.decode_ctc import get_args


def test_get_args():
    args = get_args(
        argv=[
            "syms",
            "img_list",
            "--common.checkpoint=model.ckpt",
            "--data.color_mode=RGBA",
            "--decode.use_symbols=false",
            "--decode.convert_spaces=true",
        ]
    )
    assert isinstance(args, dict)
    assert isinstance(args["decode"], DecodeArgs)
    assert args["img_list"] == "img_list"
    assert args["common"].checkpoint == "model.ckpt"
    assert args["img_dirs"] is None
    assert not args["decode"].use_symbols
    assert args["decode"].separator == " "
    assert args["decode"].join_string == " "
    assert args["decode"].convert_spaces
    assert args["data"].color_mode == "RGBA"
    assert issubclass(type(args["data"].color_mode), Enum)
    assert args["decode"].output_space == " "


@pytest.mark.parametrize(
    "arg",
    [
        None,
        "--data.batch_size=0",
        "--data.color_mode=rgb",
        "--decode.include_img_ids=t",
        "--decode.use_symbols=f",
        "--decode.convert_spaces=1",
    ],
)
def test_invalid_args(arg):
    args = [] if arg is None else ["syms", "img_list", arg]
    with pytest.raises(SystemExit):
        get_args(argv=args)


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-decode-ctc"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    help = proc.stdout.decode()
    help = " ".join(help.split())
    assert help.startswith("usage: pylaia-htr-decode-ctc")
    assert "syms img_list" in help
    assert "--common.monitor {va_loss,va_cer,va_wer}" in help
    assert "Batch size (type: PositiveInt, default: 8)" in help
    assert "--data.color_mode {L,RGB,RGBA}" in help
    assert "Decode arguments:" in help
    assert "type: Union[str, null], default: null" in help
    assert "(type: str, default: <space>)" in help
    assert "--decode.segmentation {char,word,null}" in help


expected_config = """syms: null
img_list: null
img_dirs: null
common:
  seed: 74565
  train_path: ''
  model_filename: model
  experiment_dirname: experiment
  monitor: va_cer
  checkpoint: null
data:
  batch_size: 8
  color_mode: L
logging:
  fmt: '[%(asctime)s %(levelname)s %(name)s] %(message)s'
  level: INFO
  filepath: null
  overwrite: false
  to_stderr_level: ERROR
decode:
  include_img_ids: true
  separator: ' '
  join_string: ' '
  use_symbols: true
  convert_spaces: false
  input_space: <space>
  output_space: ' '
  segmentation: null
  print_line_confidence_scores: false
  print_word_confidence_scores: false"""


def test_config_output():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-decode-ctc"), "--print_config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    config = proc.stdout.decode().strip()
    expected = expected_config + "\ntrainer:"
    assert config.startswith(expected)


def test_config_input(tmpdir):
    config = tmpdir / "config"
    config.write_text(expected_config, "utf-8")
    args = get_args(
        [f"--config={config}", "a", "b", "--img_dirs=[]", "--decode.join_string=null"]
    )
    assert args["syms"] == "a"
    assert args["img_list"] == "b"
    assert not args["img_dirs"]
    assert args["decode"].join_string is None
