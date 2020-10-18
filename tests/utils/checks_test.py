from distutils.version import StrictVersion

import pytest
import torch

import laia.common.logging as log
from laia.utils import check_inf, check_nan


@pytest.mark.parametrize(
    "dtype",
    [torch.half, torch.float, torch.double]
    if StrictVersion(torch.__version__) >= StrictVersion("1.6.0")
    else [torch.float, torch.double],
)
@pytest.mark.parametrize("raise_exception", [True, False])
def test_check_inf(caplog, dtype, raise_exception):
    tensor = torch.tensor([1, float("inf"), 3], dtype=dtype)
    log.set_level(log.DEBUG)
    if raise_exception:
        with pytest.raises(ValueError, match=r"1 \(33.33%\) INF values found"):
            check_inf(tensor, raise_exception=True)
    else:
        assert check_inf(tensor, raise_exception=False, msg="{foo} message", foo="test")
        assert caplog.messages.count("test message") == 1


@pytest.mark.parametrize(
    ["has_inf", "dtype", "log_level"],
    [
        (True, torch.float, log.INFO),
        (False, torch.int, log.DEBUG),
        (False, torch.float, log.DEBUG),
    ],
)
def test_check_inf_no_action(caplog, has_inf, dtype, log_level):
    tensor = torch.tensor([1, float("inf"), 3] if has_inf else [1, 2, 3], dtype=dtype)
    log.set_level(log_level)
    assert not check_inf(tensor, msg="test message")
    assert caplog.messages.count("test message") == 0


@pytest.mark.parametrize(
    "dtype",
    [torch.half, torch.float, torch.double]
    if StrictVersion(torch.__version__) >= StrictVersion("1.6.0")
    else [torch.float, torch.double],
)
@pytest.mark.parametrize("raise_exception", [True, False])
def test_check_nan(caplog, dtype, raise_exception):
    tensor = torch.tensor([1, float("nan"), 3], dtype=dtype)
    log.set_level(log.DEBUG)
    if raise_exception:
        with pytest.raises(ValueError, match=r"1 \(33.33%\) NaN values found"):
            check_nan(tensor, raise_exception=True)
    else:
        assert check_nan(tensor, raise_exception=False, msg="{foo} message", foo="test")
        assert caplog.messages.count("test message") == 1


@pytest.mark.parametrize(
    ["has_nan", "dtype", "log_level"],
    [
        (True, torch.float, log.INFO),
        (False, torch.int, log.DEBUG),
        (False, torch.float, log.DEBUG),
    ],
)
def test_check_nan_no_action(caplog, has_nan, dtype, log_level):
    tensor = torch.tensor([1, float("nan"), 3] if has_nan else [1, 2, 3], dtype=dtype)
    log.set_level(log_level)
    assert not check_nan(tensor, msg="test message")
    assert caplog.messages.count("test message") == 0
