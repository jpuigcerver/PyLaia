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
        "--crnn.num_input_channels=5.0",
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
    assert "(type: float v>=0 and v<=1, default: 0.5)" in help


expected_config = """adaptive_pooling: avgpool-16
common:
  checkpoint: null
  experiment_dirname: experiment
  model_filename: model
  monitor: va_cer
  seed: 74565
  train_path: ''
crnn:
  cnn_activation:
  - LeakyReLU
  - LeakyReLU
  - LeakyReLU
  - LeakyReLU
  cnn_batchnorm:
  - false
  - false
  - false
  - false
  cnn_dilation:
  - 1
  - 1
  - 1
  - 1
  cnn_dropout:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  cnn_kernel_size:
  - 3
  - 3
  - 3
  - 3
  cnn_num_features:
  - 16
  - 16
  - 32
  - 32
  cnn_poolsize:
  - 2
  - 2
  - 2
  - 0
  cnn_stride:
  - 1
  - 1
  - 1
  - 1
  lin_dropout: 0.5
  num_input_channels: 1
  rnn_dropout: 0.5
  rnn_layers: 3
  rnn_type: LSTM
  rnn_units: 256
  use_masks: false
  vertical_text: false
fixed_input_height: 0
logging:
  filepath: null
  fmt: '[%(asctime)s %(levelname)s %(name)s] %(message)s'
  level: INFO
  overwrite: false
  to_stderr_level: ERROR
save_model: true
syms: null"""


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
    config.write_text(expected_config, "utf-8")
    args = get_args([f"--config={config}", "--save_model=False"])
    assert not args["save_model"]
