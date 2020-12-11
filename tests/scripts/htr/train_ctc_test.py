from distutils.version import StrictVersion

import pytest
import torch
from conftest import call_script
from pytorch_lightning import seed_everything

from laia.common.arguments import (
    CommonArgs,
    DataArgs,
    OptimizerArgs,
    SchedulerArgs,
    TrainArgs,
    TrainerArgs,
)
from laia.common.saver import ModelSaver
from laia.dummies import DummyMNISTLines
from laia.models.htr.laia_crnn import LaiaCRNN
from laia.scripts.htr import train_ctc as script
from laia.utils import SymbolsTable


def prepare_data(dir, image_sequencer="avgpool-8"):
    seed_everything(0x12345)
    data_module = DummyMNISTLines(samples_per_space=5)
    data_module.prepare_data()
    prepare_model(dir, image_sequencer)
    # prepare syms file
    syms = dir / "syms"
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


# TODO: add ddp_cpu test
@pytest.mark.parametrize(
    "accelerator",
    [None, "ddp"] if torch.cuda.device_count() > 1 else [None],
)
def test_train_1_epoch(tmpdir, accelerator):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    # we cant just call run ourselves due to pytest-ddp issues
    args = [
        syms,
        img_dirs,
        data_module.root / "tr.gt",
        data_module.root / "va.gt",
        f"--common.train_path={tmpdir}",
        "--data.batch_size=3",
        "--train.checkpoint_k=1",
        "--trainer.max_epochs=1",
    ]
    if accelerator:
        args.append(f"--trainer.accelerator={accelerator}")
        args.append(f"--trainer.gpus=2")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stderr:\n{stderr}")

    assert not stdout
    assert "as top 1" in stderr
    assert "Saving latest checkpoint" in stderr
    assert "Best va_cer=0." in stderr
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
        img_dirs,
        data_module.root / "tr.gt",
        data_module.root / "va.gt",
        f"--common.train_path={tmpdir}",
        f"--data.batch_size=3",
        "--trainer.fast_dev_run=true",
        "--trainer.precision=16",
        "--trainer.gpus=1",
    ]
    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stderr:\n{stderr}")

    assert "Running in fast_dev_run" in stderr
    assert "Using native 16bit precision" in stderr
    assert "Model has been trained for" in stderr


def test_train_can_resume_training(tmpdir, caplog):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    caplog.set_level("INFO")
    args = [
        syms,
        img_dirs,
        data_module.root / "tr.gt",
        data_module.root / "va.gt",
    ]
    kwargs = {
        "common": CommonArgs(train_path=tmpdir),
        "data": DataArgs(batch_size=3),
        "optimizer": OptimizerArgs(name="SGD"),
        "train": TrainArgs(augment_training=True),
        "trainer": TrainerArgs(
            progress_bar_refresh_rate=0, weights_summary=None, max_epochs=1
        ),
    }
    # run to have a checkpoint
    script.run(*args, **kwargs)
    assert "Model has been trained for 1 epochs (11 steps)" in caplog.messages
    caplog.clear()

    # train for one more epoch
    kwargs["train"] = TrainArgs(resume=1)
    script.run(*args, **kwargs)
    assert "Model has been trained for 2 epochs (21 steps)" in caplog.messages


def test_train_early_stops(tmpdir, caplog):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    caplog.set_level("INFO")
    script.run(
        syms,
        img_dirs,
        data_module.root / "tr.gt",
        data_module.root / "va.gt",
        common=CommonArgs(train_path=tmpdir),
        data=DataArgs(batch_size=3),
        train=TrainArgs(early_stopping_patience=2),
        trainer=TrainerArgs(
            progress_bar_refresh_rate=0, weights_summary=None, max_epochs=5
        ),
    )
    assert (
        sum(
            m.startswith("Early stopping triggered after epoch 3 (waited for 2 epochs)")
            for m in caplog.messages
        )
        == 1
    )


def test_train_with_scheduler(tmpdir, caplog):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    caplog.set_level("INFO")
    script.run(
        syms,
        img_dirs,
        data_module.root / "tr.gt",
        data_module.root / "va.gt",
        common=CommonArgs(train_path=tmpdir),
        data=DataArgs(batch_size=3),
        optimizer=OptimizerArgs(learning_rate=1),
        scheduler=SchedulerArgs(active=True, patience=0, monitor="va_wer", factor=0.5),
        trainer=TrainerArgs(
            progress_bar_refresh_rate=0, weights_summary=None, max_epochs=5
        ),
    )
    assert "E1: lr-RMSprop 1.000e+00 ⟶ 5.000e-01" in caplog.messages
    assert "E2: lr-RMSprop 5.000e-01 ⟶ 2.500e-01" in caplog.messages


@pytest.mark.skipif(
    StrictVersion(torch.__version__) < StrictVersion("1.5.0"),
    reason="1.4.0 needs more epochs",
)
def test_train_can_overfit_one_image(tmpdir, caplog):
    syms, img_dirs, data_module = prepare_data(tmpdir)
    # manually select a specific image
    txt_file = data_module.root / "tr.gt"
    line = "tr-6 9 2 0 1"
    assert txt_file.read_text().splitlines()[6] == line
    txt_file.write_text(line)

    caplog.set_level("INFO")
    script.run(
        syms,
        img_dirs,
        txt_file,
        txt_file,
        common=CommonArgs(
            train_path=tmpdir, seed=0x12345, experiment_dirname="", monitor="va_loss"
        ),
        data=DataArgs(batch_size=1),
        # after some manual runs, this lr seems to be the
        # fastest one to reliably learn for this toy example.
        # RMSProp performed considerably better than Adam|SGD
        optimizer=OptimizerArgs(learning_rate=0.01, name="RMSProp"),
        train=TrainArgs(
            checkpoint_k=0,  # disable checkpoints
            early_stopping_patience=100,  # disable early stopping
        ),
        trainer=TrainerArgs(
            weights_summary=None,
            overfit_batches=1,
            max_epochs=70,
            check_val_every_n_epoch=100,  # disable validation
        ),
    )
    assert sum("cer=0.0%" in m and "wer=0.0%" in m for m in caplog.messages)


def test_raises(tmpdir):
    with pytest.raises(AssertionError, match="Could not find the model"):
        script.run("", [], "", "")

    syms, img_dirs, data_module = prepare_data(tmpdir)
    with pytest.raises(AssertionError, match='The delimiter "TEST" is not available'):
        script.run(
            syms,
            [],
            "",
            "",
            common=CommonArgs(train_path=tmpdir),
            train=TrainArgs(delimiters=["<space>", "TEST"]),
        )
