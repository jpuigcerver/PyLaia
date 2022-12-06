import pytest
import torch
from conftest import call_script
from pytorch_lightning import seed_everything

from laia.common.arguments import CommonArgs
from laia.common.saver import ModelSaver
from laia.dummies import DummyMNISTLines, DummyModel
from laia.scripts.htr import netout as script


# TODO: fix test with nprocs=2
@pytest.mark.parametrize("nprocs", (1,))
def test_netout_on_dummy_mnist_lines_data(tmpdir, nprocs):
    seed_everything(0x12345)
    # prepare data
    data_module = DummyMNISTLines(tr_n=0, va_n=5, batch_size=3, samples_per_space=3)
    data_module.prepare_data()
    # prepare model file
    final_size, classes = 3, 12  # 12 == 10 digits + space + ctc
    model_args = [(final_size,) * 2, classes]
    ModelSaver(tmpdir).save(DummyModel, *model_args)
    # prepare ckpt file
    ckpt = tmpdir / "model.ckpt"
    torch.save(DummyModel(*model_args).state_dict(), str(ckpt))
    # prepare img list
    img_list = tmpdir / "img_list"
    img_list.write_text(
        "\n".join(f"va-{i}" for i in range(data_module.n["va"])), "utf-8"
    )

    args = [
        img_list,
        f"--img_dirs=[{str(data_module.root / 'va')}]",
        f"--common.train_path={tmpdir}",
        f"--common.checkpoint={ckpt}",
        f"--common.experiment_dirname={tmpdir}",
        f"--data.batch_size={data_module.batch_size}",
        "--netout.output_transform=softmax",
        "--netout.digits=3",
        "--netout.lattice=lattice",
        "--netout.matrix=matrix",
    ]
    if nprocs > 1:
        args.append("--trainer.accelerator=ddp_cpu")
        args.append(f"--trainer.num_processes={nprocs}")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    assert "Using checkpoint" in stderr

    lattice = tmpdir / "lattice"
    assert lattice.exists()
    lines = [l.strip() for l in lattice.read_text("utf-8").split("\n") if l]
    n = data_module.n["va"]
    assert len(lines) == n * (final_size * classes + 1) + n

    # this is harder to test so do some basic checks
    matrix = tmpdir / "matrix"
    assert matrix.exists()
    assert len(matrix.read_binary()) > 0


def test_raises(tmpdir):
    # generate a model and a checkpoint
    model_args = [(1, 1), 1]
    ModelSaver(tmpdir).save(DummyModel, *model_args)
    ckpt = tmpdir / "model.ckpt"
    torch.save(DummyModel(*model_args).state_dict(), str(ckpt))

    with pytest.raises(AssertionError, match="Could not find the model"):
        script.run(
            "",
            common=CommonArgs(
                train_path=tmpdir,
                experiment_dirname="",
                model_filename="test",
                checkpoint="model.ckpt",
            ),
        )

    with pytest.raises(AssertionError, match="You did not specify any output file"):
        script.run(
            "",
            common=CommonArgs(
                train_path=tmpdir, experiment_dirname="", checkpoint="model.ckpt"
            ),
        )
