import shutil
import subprocess

import pytest

from laia.common.arguments import CommonArgs, TrainerArgs
from laia.scripts.htr.train_ctc import get_args


def test_get_args(tmpdir):
    cmd_args = [
        "syms",
        "[tr,va,te]",
        "tr_table",
        "va_table",
        "--train.delimiters=null",
        "--train.checkpoint_k=-1",
        "--scheduler.monitor=va_wer",
        "--scheduler.patience=20",
        "--trainer.gpus=[1,2]",
    ]
    args = get_args(argv=cmd_args)
    assert isinstance(args, dict)
    assert isinstance(args["common"], CommonArgs)
    assert isinstance(args["trainer"], TrainerArgs)
    assert args["img_dirs"] == ["tr", "va", "te"]
    assert args["tr_txt_table"] == "tr_table"
    assert args["va_txt_table"] == "va_table"
    assert not args["train"].delimiters
    assert args["train"].checkpoint_k == -1
    assert args["scheduler"].monitor == "va_wer"
    assert args["scheduler"].patience == 20
    assert args["trainer"].gpus == [1, 2]


@pytest.mark.parametrize(
    "arg",
    [
        "--train.checkpoint_k=-2",
        "--train.resume=f",
        "--train.early_stopping_patience=-1",
        "--train.gpu_stats=y",
        "--train.augment_training=1",
        "--optimizer.name=AdamW",
        "--optimizer.name=ADAM",
        "--optimizer.learning_rate=0.0",
        "--optimizer.momentum=-1",
        "--optimizer.nesterov=",
    ],
)
def test_invalid_args(arg):
    args = ["syms", "[img_dir]", "tr_txt", "va_txt", arg]
    with pytest.raises(SystemExit):
        get_args(argv=args)


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-train-ctc"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    help = proc.stdout.decode()
    help = " ".join(help.split())
    assert help.startswith("usage: pylaia-htr-train-ctc")
    assert "syms img_dirs tr_txt_table va_txt_table" in help
    assert "Mapping from strings to integers" in help
    assert "--common.experiment_dirname EXPERIMENT_DIRNAME" in help
    assert "Any of: 1" in help
    assert "(type: Union[List[str], null], default: ['<space>'])" in help
    assert "(type: int_ge-1, default: 3)" in help
    assert "--train.resume RESUME" in help
    assert "Union[bool, NonNegativeInt]" in help
    assert "[%(asctime)s %(levelname)s %(name)s] %(message)s" in help
    assert "--optimizer.name {SGD,RMSProp,Adam}" in help
    assert "(type: Monitor, default: va_loss)" in help


expected_config = """syms: null
img_dirs: []
tr_txt_table: null
va_txt_table: null
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
train:
  delimiters:
  - <space>
  checkpoint_k: 3
  resume: false
  early_stopping_patience: 20
  gpu_stats: false
  augment_training: false
logging:
  fmt: '[%(asctime)s %(levelname)s %(name)s] %(message)s'
  level: INFO
  filepath: null
  overwrite: false
  to_stderr_level: ERROR
optimizer:
  name: RMSProp
  learning_rate: 0.0005
  momentum: 0.0
  weight_l2_penalty: 0.0
  nesterov: false
scheduler:
  active: false
  monitor: va_loss
  patience: 5
  factor: 0.1"""


def test_config_output():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-train-ctc"), "--print_config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    config = proc.stdout.decode().strip()
    expected = expected_config + "\ntrainer:"
    assert config.startswith(expected)


def test_config_input(tmpdir):
    config = tmpdir / "config"
    config.write_text(expected_config, "utf-8")
    args = get_args([f"--config={config}", "a", "[b,c]", "d", "e"])
    assert args["syms"] == "a"
    assert args["img_dirs"] == ["b", "c"]
    assert args["tr_txt_table"] == "d"
    assert args["va_txt_table"] == "e"
