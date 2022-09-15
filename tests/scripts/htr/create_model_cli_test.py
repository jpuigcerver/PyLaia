import shutil
import subprocess

import pytest

from laia.common.arguments import CreateCRNNArgs
from laia.scripts.htr.create_model import get_args


@pytest.mark.parametrize(
    "cmd_args",
    [
        [],
        # model used to train IAM-htr
        [
            "--adaptive_pooling=avgpool-16",
            "--fixed_input_height=128",
            "--crnn.cnn_num_features=[16,32,48,64,80]",
            "--crnn.cnn_kernel_size=[3,3,3,3,3]",
            "--crnn.cnn_stride=[1,1,1,1,1]",
            "--crnn.cnn_dilation=[1,1,1,1,1]",
            f"--crnn.cnn_activation=[{','.join(['LeakyReLU'] * 5)}]",
            "--crnn.cnn_poolsize=[2,2,2,0,0]",
            "--crnn.cnn_dropout=[0,0,0,0,0]",
            "--crnn.cnn_batchnorm=[false,false,false,false,false]",
            "--crnn.rnn_units=256",
            "--crnn.rnn_layers=5",
        ],
    ],
)
def test_get_args(cmd_args):
    args = get_args(argv=["syms"] + cmd_args)
    assert isinstance(args, dict)
    assert isinstance(args["crnn"], CreateCRNNArgs)
    assert args["syms"] == "syms"


@pytest.mark.parametrize(
    "arg",
    [
        None,
        "--fixed_input_height=-1",
        "--save_model=t",
        "--common.monitor=va.xer",
        "--logging.level=10",
        "--logging.level=warning",
        "--crnn.num_input_channels=a",
        "--crnn.num_input_channels=-5",
        "--crnn.cnn_num_features=[10,0,10]",
        "--crnn.rnn_dropout=-0.5",
    ],
)
def test_invalid_args(arg):
    args = [] if arg is None else ["syms", arg]
    with pytest.raises(SystemExit):
        get_args(argv=args)


def test_entry_point():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-create-model"), "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    help = proc.stdout.decode()
    help = " ".join(help.split())
    assert help.startswith("usage: pylaia-htr-create-model")
    assert "mapped to integer 0 (required, type: str)" in help
    assert "--common.monitor {va_loss,va_cer,va_wer}" in help
    assert "Create LaiaCRNN arguments:" in help
    assert "--logging.level {NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL}" in help
    assert "Number of channels of the input images" in help
    assert (
        "(type: List[Union[PositiveInt, List[PositiveInt]]], default: [3, 3, 3, 3])"
        in help
    )
    assert "(type: List[PositiveInt], default: [16, 16, 32, 32])" in help
    assert "--crnn.lin_dropout LIN_DROPOUT" in help
    assert "(type: ClosedUnitInterval, default: 0.5)" in help


expected_config = """syms: null
fixed_input_height: 0
adaptive_pooling: avgpool-16
save_model: true
common:
  seed: 74565
  train_path: ''
  model_filename: model
  experiment_dirname: experiment
  monitor: va_cer
  checkpoint: null
logging:
  fmt: '[%(asctime)s %(levelname)s %(name)s] %(message)s'
  level: INFO
  filepath: null
  overwrite: false
  to_stderr_level: ERROR
crnn:
  num_input_channels: 1
  vertical_text: false
  cnn_num_features:
  - 16
  - 16
  - 32
  - 32
  cnn_kernel_size:
  - 3
  - 3
  - 3
  - 3
  cnn_stride:
  - 1
  - 1
  - 1
  - 1
  cnn_dilation:
  - 1
  - 1
  - 1
  - 1
  cnn_activation:
  - LeakyReLU
  - LeakyReLU
  - LeakyReLU
  - LeakyReLU
  cnn_poolsize:
  - 2
  - 2
  - 2
  - 0
  cnn_dropout:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  cnn_batchnorm:
  - false
  - false
  - false
  - false
  use_masks: false
  rnn_layers: 3
  rnn_units: 256
  rnn_dropout: 0.5
  rnn_type: LSTM
  lin_dropout: 0.5"""

expected_syms = """<ctc> 0
a 1
b 2
c 3"""


def test_config_output():
    proc = subprocess.run(
        [shutil.which("pylaia-htr-create-model"), "--print_config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    config = proc.stdout.decode().strip()
    assert config == expected_config


def test_config_input(tmpdir):
    config = tmpdir / "config"
    syms = (
        tmpdir / "syms"
    )  # create dummy syms.txt to avoid TypeError: Configuration check failed

    config.write_text(expected_config.replace("syms: null", f"syms: {syms}"), "utf-8")
    syms.write_text(expected_syms, "utf-8")

    args = get_args(argv=[f"--config={config}", "--save_model=False"])
    assert not args["save_model"]
