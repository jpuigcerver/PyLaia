import shutil
import subprocess

import pytest

from laia.common.arguments import NetoutArgs
from laia.scripts.htr.netout import get_args


def test_get_args():
    args = get_args(argv=["img_list"])
    assert isinstance(args, dict)
    assert isinstance(args["netout"], NetoutArgs)
    assert args["img_list"] == "img_list"
    assert args["img_dirs"] is None
    assert args["netout"].matrix is args["netout"].lattice is None
    assert isinstance(args["netout"].digits, int)


@pytest.mark.parametrize(
    "arg",
    [None, "--netout.digits=5.0", "--netout.digits=-1"],
)
def test_invalid_args(arg):
    args = [] if arg is None else ["img_list", arg]
    with pytest.raises(SystemExit):
        get_args(argv=args)


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-netout"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    help = proc.stdout.decode()
    help = " ".join(help.split())
    assert help.startswith("usage: pylaia-htr-netout")
    assert "File containing the images to decode" in help
    assert " --netout.output_transform {softmax,log_softmax,null}" in help
    assert "(type: Union[str, null], default: null)" in help
    assert "containing the output matrices" in help
    assert "containing the output lattices" in help
    assert "used for formatting (type: int v>=0, default: 10)" in help


expected_config = """common:
  checkpoint: null
  experiment_dirname: experiment
  model_filename: model
  monitor: va_cer
  seed: 74565
  train_path: ''
data:
  batch_size: 8
  color_mode: L
img_dirs: null
img_list: null
logging:
  filepath: null
  fmt: '[%(asctime)s %(levelname)s %(name)s] %(message)s'
  level: INFO
  overwrite: false
  to_stderr_level: ERROR
netout:
  digits: 10
  lattice: null
  matrix: null
  output_transform: null"""


def test_config_output():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-netout"), "--print-config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    config = proc.stdout.decode().strip()
    expected = expected_config + "\ntrainer:"
    assert config.startswith(expected)


def test_config_input(tmpdir):
    config = tmpdir / "config"
    config.write_text(expected_config, "utf-8")
    args = get_args([f"--config={config}", "a", "--img_dirs=[]"])
    assert args["img_list"] == "a"
    assert not len(args["img_dirs"])
