import shutil
from os.path import join

import pytest
import torch
from pytorch_lightning import seed_everything

from laia.common.saver import ModelSaver
from laia.dummies import DummyMNISTLines, DummyModel
from laia.scripts.htr import decode_ctc as script
from laia.utils import SymbolsTable
from tests.scripts.htr.script_utils import call_script, downloader  # noqa


@pytest.mark.parametrize("nprocs", (1, 2))
def test_decode_on_dummy_mnist_lines_data(tmpdir, nprocs):
    seed_everything(0x12345)
    # prepare data
    data_module = DummyMNISTLines(tr_n=0, va_n=5, batch_size=3, samples_per_space=3)
    data_module.prepare_data()
    # prepare model file
    model_args = [(3, 3), 12]
    ModelSaver(tmpdir).save(DummyModel, *model_args)
    # prepare ckpt file
    ckpt = tmpdir / "model.ckpt"
    torch.save(DummyModel(*model_args).state_dict(), str(ckpt))
    # prepare syms file
    syms = str(tmpdir / "syms")
    syms_table = SymbolsTable()
    for k, v in data_module.syms.items():
        syms_table.add(v, k)
    syms_table.save(syms)
    # prepare img list
    img_list = tmpdir / "img_list"
    img_list.write_text(
        "\n".join(f"va-{i}" for i in range(data_module.n["va"])), "utf-8"
    )

    args = [
        syms,
        str(img_list),
        ckpt,
        join(data_module.root, "va"),
        f"--train_path={tmpdir}",
        f"--experiment_dirname={tmpdir}",
        f"--batch_size={data_module.batch_size}",
    ]
    if nprocs > 1:
        args.append("--accelerator=ddp_cpu")
        args.append(f"--num_processes={nprocs}")

    stdout, stderr = call_script(script.__file__, args)

    img_ids = [l.split(" ", maxsplit=1)[0] for l in stdout.strip().split("\n")]
    assert sorted(img_ids) == [f"va-{i}" for i in range(data_module.n["va"])]
    assert "Using checkpoint" in stderr


@pytest.mark.parametrize("nprocs", (1, 2))
def test_decode_with_trained_ckpt_fixed_height(tmpdir, downloader, nprocs):
    syms = downloader("print/syms.txt")
    img_list = downloader("print/imgs.lst")
    ckpt = downloader("print/experiment_h128")
    images = downloader("print/imgs_h128", archive=True)
    model = downloader("print/model_h128")
    shutil.move(model, tmpdir)

    args = [
        syms,
        img_list,
        ckpt,
        images,
        f"--train_path={tmpdir}",
        f"--experiment_dirname={tmpdir}",
        "--model_filename=model_h128",
        f"--batch_size=3",
        "--join_str=",
        "--convert_spaces",
    ]
    if nprocs > 1:
        args.append("--accelerator=ddp_cpu")
        args.append(f"--num_processes={nprocs}")

    stdout, _ = call_script(script.__file__, args)

    lines = sorted(stdout.strip().split("\n"))
    assert lines == [
        "ONB_aze_18950706_4.r_10_2.tl_125 Deutschland.",
        "ONB_aze_18950706_4.r_10_3.tl_126 — Wir haben gestern von dem merkwürdigen Tadel",
        "ONB_aze_18950706_4.r_10_3.tl_127 erzahlt, den der Colberger Bürgermeister von dem vorgesetzten",
        "ONB_aze_18950706_4.r_10_3.tl_128 Regierungspräsidenten in Köslin erlitten hat, weil er anläß¬",
        "ONB_aze_18950706_4.r_10_3.tl_129 lich der Reichsrathswahl im Kreise Colberg=Köslin den",
    ]


def test_decode_with_old_trained_ckpt(tmpdir, downloader):
    syms = downloader("print/syms.txt")
    img_list = downloader("print/imgs.lst")
    ckpt = downloader("print/old_experiment")
    images = downloader("print/imgs", archive=True)
    # download and move model
    model = downloader("print/old_model")
    shutil.move(model, tmpdir)

    args = [
        syms,
        img_list,
        ckpt,
        images,
        f"--train_path={tmpdir}",
        f"--experiment_dirname={tmpdir}",
        "--model_filename=old_model",
        f"--batch_size=3",
        "--join_str=",
        "--convert_spaces",
    ]
    stdout, _ = call_script(script.__file__, args)

    lines = sorted(stdout.strip().split("\n"))
    assert lines == [
        "ONB_aze_18950706_4.r_10_2.tl_125 Deuklichland.",
        "ONB_aze_18950706_4.r_10_3.tl_126 — Wir haben gestern von dem merkwürdigen Tadel",
        "ONB_aze_18950706_4.r_10_3.tl_127 erzählt, den der Colberger Bürgermeister von dem vorgesetzten",
        "ONB_aze_18950706_4.r_10_3.tl_128 Regierungspräsidenten in Köslin erlitten hat, weil er anläß¬",
        "ONB_aze_18950706_4.r_10_3.tl_129 lich der Reicherathswahl in Kreise Colberg=Köslin den",
    ]
