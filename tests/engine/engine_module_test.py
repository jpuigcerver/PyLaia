from logging import DEBUG

import pytest
import torch

from laia.common.arguments import OptimizerArgs, SchedulerArgs
from laia.dummies import DummyMNIST, DummyModel, DummyTrainer
from laia.engine import EngineModule
from laia.engine.engine_exception import EngineException
from laia.losses import CTCLoss


@pytest.mark.parametrize(
    ["name", "expected"],
    [
        ("SGD", torch.optim.SGD),
        ("RMSProp", torch.optim.RMSprop),
        ("Adam", torch.optim.Adam),
    ],
)
def test_configure_optimizers(name, expected):
    model = DummyModel((3, 3), 10)
    optimizer = EngineModule(
        model, lambda x: x, optimizer=OptimizerArgs(name=name)
    ).configure_optimizers()
    assert isinstance(optimizer, expected)


def test_configure_optimizers_scheduler():
    model = DummyModel((3, 3), 10)
    opt, sch = EngineModule(
        model,
        lambda x: x,
        OptimizerArgs(name="SGD"),
        scheduler=SchedulerArgs(active=True),
    ).configure_optimizers()
    assert isinstance(opt[0], torch.optim.SGD)
    assert isinstance(sch[0]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_run_checks(caplog):
    model = DummyModel((3, 3), 10)
    module = EngineModule(model, lambda x: x)
    with caplog.at_level(DEBUG):
        module.check_tensor(None, torch.tensor([1, 5, float("nan"), float("-inf"), -5]))
    assert caplog.messages == [
        "Found 2 (40.00%) infinite values in the model output at epoch=0, batch=None, global_step=0",
    ]


def test_exception_catcher():
    model = DummyModel((3, 3), 10)
    module = EngineModule(model, lambda x: x)
    with pytest.raises(EngineException, match=r'Exception "RuntimeError\(\)" raised'):
        with module.exception_catcher(None):
            raise RuntimeError()


def test_compute_loss():
    model = DummyModel((3, 3), 10)

    with pytest.raises(
        EngineException,
        match=r'Exception "ValueError\(\'The loss is NaN or ± inf\'[,]?\)" raised',
    ):
        module = EngineModule(model, lambda *_: torch.tensor([1, float("nan")]))
        module.compute_loss(None, None, None)

    expected = torch.tensor(0.1)
    module = EngineModule(model, lambda *_: expected)
    actual = module.compute_loss(None, None, None)
    torch.testing.assert_allclose(actual, expected)


def test_can_train(tmpdir):
    model = DummyModel((3, 3), 10)
    module = EngineModule(model, CTCLoss())
    trainer = DummyTrainer(default_root_dir=tmpdir)
    trainer.fit(module, datamodule=DummyMNIST())
