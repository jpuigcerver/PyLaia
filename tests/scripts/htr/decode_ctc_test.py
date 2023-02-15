import os
import shutil
from io import StringIO
from unittest import mock

import pytest
import torch
from conftest import call_script
from packaging import version
from pytorch_lightning import seed_everything

from laia.common.arguments import CommonArgs, DataArgs, DecodeArgs
from laia.common.saver import ModelSaver
from laia.dummies import DummyMNISTLines, DummyModel
from laia.scripts.htr import decode_ctc as script
from laia.utils import SymbolsTable


# TODO: fix test with nprocs=2
# @pytest.mark.parametrize("nprocs", (1,2))
@pytest.mark.parametrize("nprocs", (1,))
def test_decode_on_dummy_mnist_lines_data(tmpdir, nprocs):
    # prepare data
    seed_everything(0x12345)
    data_module = DummyMNISTLines(tr_n=0, va_n=5, batch_size=3, samples_per_space=3)
    data_module.prepare_data()
    # prepare model file
    model_args = [(3, 3), 12]
    ModelSaver(tmpdir).save(DummyModel, *model_args)
    # prepare ckpt file
    ckpt = tmpdir / "model.ckpt"
    torch.save(DummyModel(*model_args).state_dict(), str(ckpt))
    # prepare syms file
    syms = tmpdir / "syms"
    data_module.syms.save(syms)
    # prepare img list
    img_list = tmpdir / "img_list"
    img_list.write_text(
        "\n".join(f"va-{i}" for i in range(data_module.n["va"])), "utf-8"
    )

    args = [
        syms,
        img_list,
        f"--img_dirs={[str(data_module.root / 'va')]}",
        f"--common.checkpoint={ckpt}",
        f"--common.train_path={tmpdir}",
        f"--data.batch_size={data_module.batch_size}",
    ]
    if nprocs > 1:
        args.append("--trainer.accelerator=ddp_cpu")
        args.append(f"--trainer.num_processes={nprocs}")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    img_ids = [l.split(" ", maxsplit=1)[0] for l in stdout.strip().split("\n")]
    assert sorted(img_ids) == [f"va-{i}" for i in range(data_module.n["va"])]
    assert "Using checkpoint" in stderr


# @pytest.mark.parametrize(
#     "accelerator",
#     [None, "ddp_cpu", "ddp"] if torch.cuda.device_count() > 1 else [None, "ddp_cpu"],
# )
@pytest.mark.parametrize(
    "accelerator",
    [None, "ddp"] if torch.cuda.device_count() > 1 else [None],
)
def test_decode_with_trained_ckpt_fixed_height(downloader, accelerator):
    syms = downloader("syms.txt")
    ckpt = os.path.basename(downloader("weights.ckpt"))
    model = os.path.basename(downloader("model"))

    base_path = "tests/resources/"
    img_list = os.path.join(base_path, "img_list.txt")

    args = [
        syms,
        img_list,
        f"--img_dirs={[base_path]}",
        f"--common.train_path={base_path}",
        f"--common.checkpoint={ckpt}",
        f"--common.model_filename={model}",
        "--data.batch_size=3",
        "--decode.join_string=",
        "--decode.convert_spaces=true",
    ]
    if accelerator:
        args.append(f"--trainer.accelerator={accelerator}")
        args.append(
            f"--trainer.{'num_processes' if accelerator == 'ddp_cpu' else 'gpus'}=2"
        )

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    lines = sorted(stdout.strip().split("\n"))
    assert lines == [
        "image_01.jpg dig, saasnart jeg fik ted, men nu kom dette idag,",
        "image_02.jpg – Jeg frygter for Anstrærgelse",
        "image_03.jpg Forstørrelse af et mindre og meget",
        "image_04.jpg Lov kommet friske hjem fra Holmenkol-",
        "image_05.jpg da han neppe interesseren",
    ]


def test_decode_with_old_trained_ckpt(tmpdir, downloader):
    syms = downloader("syms.txt")
    ckpt = os.path.basename(downloader("weights.ckpt"))
    model = os.path.basename(downloader("model"))

    base_path = "tests/resources/"
    img_list = os.path.join(base_path, "img_list.txt")

    stdout = StringIO()
    with mock.patch("sys.stdout", new=stdout):
        script.run(
            syms,
            img_list,
            [base_path],
            common=CommonArgs(
                train_path=base_path, checkpoint=ckpt, model_filename=model
            ),
            data=DataArgs(batch_size=3),
            decode=DecodeArgs(join_string="", convert_spaces=True),
        )
    assert sorted(stdout.getvalue().strip().split("\n")) == [
        "image_01.jpg dig, saasnart jeg fik ted, men nu kom dette idag,",
        "image_02.jpg – Jeg frygter for Anstrærgelse",
        "image_03.jpg Forstørrelse af et mindre og meget",
        "image_04.jpg Lov kommet friske hjem fra Holmenkol-",
        "image_05.jpg da han neppe interesseren",
    ]


# @pytest.mark.parametrize(
#     "accelerator",
#     [None, "ddp_cpu", "ddp"] if torch.cuda.device_count() > 1 else [None, "ddp_cpu"],
# )
@pytest.mark.parametrize(
    "accelerator",
    [None, "ddp"] if torch.cuda.device_count() > 1 else [None],
)
def test_segmentation(downloader, accelerator):
    syms = downloader("syms.txt")
    ckpt = os.path.basename(downloader("weights.ckpt"))
    model = os.path.basename(downloader("model"))

    base_path = "tests/resources/"
    img_list = os.path.join(base_path, "img_list.txt")

    args = [
        syms,
        img_list,
        f"--img_dirs={[base_path]}",
        f"--common.train_path={base_path}",
        f"--common.checkpoint={ckpt}",
        f"--common.model_filename={model}",
        "--data.batch_size=3",
        "--decode.join_string=",
        "--decode.segmentation=word",
    ]

    if accelerator:
        args.append(f"--trainer.accelerator={accelerator}")
        args.append(
            f"--trainer.{'num_processes' if accelerator == 'ddp_cpu' else 'gpus'}=2"
        )

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    lines = sorted(stdout.strip().split("\n"))
    expected = [
        "image_01.jpg [('dig,', 1, 1, 143, 128)",
        "image_02.jpg [('–', 1, 1, 39, 128)",
        "image_03.jpg [('Forstørrelse', 1, 1, 431, 128)",
        "image_04.jpg [('Lov', 1, 1, 119, 128)",
        "image_05.jpg [('da', 1, 1, 127, 128)",
    ]
    assert all(l.startswith(e) for l, e in zip(lines, expected))


def test_raises(tmpdir):
    # generate a model and a checkpoint
    model_args = [(1, 1), 1]
    ModelSaver(tmpdir).save(DummyModel, *model_args)
    ckpt = tmpdir / "model.ckpt"
    torch.save(DummyModel(*model_args).state_dict(), str(ckpt))

    with pytest.raises(AssertionError, match="Could not find the model"):
        script.run(
            "",
            "",
            common=CommonArgs(
                train_path=tmpdir,
                experiment_dirname="",
                model_filename="test",
                checkpoint="model.ckpt",
            ),
        )
