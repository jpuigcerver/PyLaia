from distutils.version import StrictVersion

import pytest
import torch
from pytorch_lightning import seed_everything

from laia.common.saver import ModelSaver
from laia.dummies import DummyMNISTLines
from laia.models.htr.laia_crnn import LaiaCRNN
from laia.scripts.htr import train_ctc as script
from laia.utils import SymbolsTable
from tests.scripts.htr.conftest import call_script


def prepare_data(dir, image_sequencer="avgpool-8"):
    seed_everything(0x12345)
    data_module = DummyMNISTLines(batch_size=3, samples_per_space=5)
    data_module.prepare_data()
    prepare_model(dir, image_sequencer)
    # prepare syms file
    syms = str(dir / "syms")
    syms_table = SymbolsTable()
    for k, v in data_module.syms.items():
        syms_table.add(v, k)
    syms_table.save(syms)
    # prepare img dirs
    img_dirs = [str(data_module.root / p) for p in ("tr", "va")]
    return syms, img_dirs, data_module


def prepare_model(dir, image_sequencer):
    args = [
        1,  # num_input_channels
        12,  # num_output_channels
        [16],  # cnn_num_features
        [(3, 3)],  # cnn_kernel_size
        [(1, 1)],  # cnn_stride
        [(1, 1)],  # cnn_dilation
        [torch.nn.LeakyReLU],  # cnn_activation
        [(2, 2)],  # cnn_poolsize
        [0],  # cnn_dropout
        [False],  # cnn_batchnorm
        image_sequencer,
        16,  # rnn_units
        1,  # rnn_layers
        0,  # rnn_dropout
        0,  # lin_dropout
    ]
    LaiaCRNN(*args)  # check for failures
    ModelSaver(dir).save(LaiaCRNN, *args)


# TODO: add distributed tests
@pytest.mark.parametrize("nprocs", (1,))
def test_train_fast_dev(tmpdir, nprocs):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    args = [
        syms,
        *img_dirs,
        str(data_module.root / "tr.gt"),
        str(data_module.root / "va.gt"),
        f"--train_path={tmpdir}",
        f"--batch_size={data_module.batch_size}",
        "--fast_dev_run=1",
        "--checkpoint_k=1",
    ]
    if nprocs > 1:
        args.append("--accelerator=ddp_cpu")
        args.append(f"--num_processes={nprocs}")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stderr:\n{stderr}")

    assert not len(stdout)
    assert "Running in fast_dev_run" in stderr
    assert "as top 1" in stderr
    assert "Saving latest checkpoint" in stderr
    assert "Best va_cer=0.9740259647369385" in stderr
    assert {f.basename for f in tmpdir.join("experiment").listdir()} == {
        "epoch=0-lowest_va_cer.ckpt",
        "epoch=0-last.ckpt",
        "metrics.csv",
    }


@pytest.mark.skipif(
    StrictVersion(torch.__version__) < StrictVersion("1.7.0"),
    reason="Some ops do not support AMP before 1.7.0",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP needs CUDA")
def test_train_half_precision(tmpdir):
    # TODO: add test using nnutils: https://github.com/jpuigcerver/nnutils/issues/4
    syms, img_dirs, data_module = prepare_data(tmpdir, image_sequencer="none-14")
    args = [
        syms,
        *img_dirs,
        str(data_module.root / "tr.gt"),
        str(data_module.root / "va.gt"),
        f"--train_path={tmpdir}",
        f"--batch_size={data_module.batch_size}",
        "--fast_dev_run=1",
        "--precision=16",
        "--gpus=1",
    ]
    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stderr:\n{stderr}")

    assert "Using native 16bit precision" in stderr
    assert "Model has been trained for" in stderr


def test_train_can_resume_training(tmpdir):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    args = [
        syms,
        *img_dirs,
        str(data_module.root / "tr.gt"),
        str(data_module.root / "va.gt"),
        f"--train_path={tmpdir}",
        f"--batch_size={data_module.batch_size}",
        "--progress_bar_refresh_rate=0",
        "--use_distortions",
        "--optimizer=SGD",
    ]
    # run to have a checkpoint
    _, stderr = call_script(script.__file__, args + ["--max_epochs=1"])
    print(f"Script 1 stderr:\n{stderr}")
    assert "Model has been trained for 1 epochs (11 steps)" in stderr
    # resume training
    _, stderr = call_script(
        script.__file__, args + ["--max_epochs=2", "--resume_training"]
    )
    print(f"Script 2 stderr:\n{stderr}")
    assert "Model has been trained for 2 epochs (21 steps)" in stderr


def test_train_early_stops(tmpdir):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    args = [
        syms,
        *img_dirs,
        str(data_module.root / "tr.gt"),
        str(data_module.root / "va.gt"),
        f"--train_path={tmpdir}",
        f"--batch_size={data_module.batch_size}",
        "--progress_bar_refresh_rate=0",
        "--max_epochs=5",
        "--early_stopping_patience=2",
    ]
    _, stderr = call_script(script.__file__, args)
    print(f"Script stderr:\n{stderr}")
    assert "after epoch 3 (waited for 2 epochs)" in stderr


def test_train_with_scheduler(tmpdir):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    args = [
        syms,
        *img_dirs,
        str(data_module.root / "tr.gt"),
        str(data_module.root / "va.gt"),
        f"--train_path={tmpdir}",
        f"--batch_size={data_module.batch_size}",
        "--progress_bar_refresh_rate=0",
        "--max_epochs=3",
        "--scheduler",
        "--scheduler_patience=0",
        "--scheduler_monitor=va_wer",
        "--scheduler_factor=0.5",
        "--learning_rate=1",
    ]
    _, stderr = call_script(script.__file__, args)
    print(f"Script stderr:\n{stderr}")
    assert "Epoch 1: lr-RMSprop 1.000e+00 ⟶ 5.000e-01" in stderr
    assert "Epoch 2: lr-RMSprop 5.000e-01 ⟶ 2.500e-01" in stderr
