from typing import Optional

import torch

import laia.common.logging as log


def check_tensor(
    tensor: torch.Tensor,
    msg: Optional[str] = None,
    name: Optional[str] = None,
    raise_exception: bool = False,
    **kwargs,
) -> bool:
    """
    Checks if each element of a tensor is finite or not.
    Real values are finite when they are not NaN, negative infinity, or infinity.

    Arguments:
      tensor (torch.Tensor): tensor to check.
      msg (str): message format string. The message format can use the keys
          ``abs_num`` and ``rel_num`` to print the absolute number and the
           percentage of infinite elements. (Default: None)
      name (str): Name of the logger used to log the event (Default: None)
      raise_exception (bool): raise an exception instead of logging the event
          (Default: False)
      kwargs: additional named arguments passed to format the message.

    Return:
      `True` if the tensor contains any infinite value, `False` otherwise.
    """
    logger = log.get_logger(name)
    if logger.isEnabledFor(log.DEBUG):
        num = torch.isfinite(tensor).logical_not().sum().item()
        if num > 0:
            percentage = num / tensor.numel()
            msg = (
                f"{num:d} ({percentage:.2%}) infinite values found"
                if msg is None
                else msg.format(abs_num=num, rel_num=percentage, **kwargs)
            )
            if raise_exception:
                raise ValueError(msg)
            logger.debug(msg)
            return True
    return False
