import torch

import laia.common.logging as log

_TENSOR_REAL = (torch.float, torch.double, torch.half)


def check_inf(tensor, msg=None, name=None, raise_exception=False, **kwargs):
    """Check whether a tensor contains a +/- infinite value.

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
      `True` if the tensor contains any +/- infinite element, or `False`
      otherwise.
    """
    logger = log.get_logger(name)
    if logger.isEnabledFor(log.DEBUG) and tensor.dtype in _TENSOR_REAL:
        num_inf = torch.isinf(tensor).sum().item()
        if num_inf > 0:
            per_inf = num_inf / tensor.numel()
            msg = (
                "{:d} ({:.2%}) INF values found".format(num_inf, per_inf)
                if msg is None
                else msg.format(abs_num=num_inf, rel_num=per_inf, **kwargs)
            )
            if raise_exception:
                raise ValueError(msg)
            else:
                logger.debug(msg)
            return True
    return False


def check_nan(tensor, msg=None, name=None, raise_exception=False, **kwargs):
    """Check whether a tensor contains a NaN value.

    Arguments:
      tensor (torch.Tensor): tensor to check.
      msg (str): message format string. The message format can use the keys
          ``abs_num`` and ``rel_num`` to print the absolute number and the
           percentage of NaN elements. (Default: None)
      name (str): Name of the logger used to log the event (Default: None)
      raise_exception (bool): raise an exception instead of logging the event
          (Default: False)
      kwargs: additional named arguments passed to format the message.

    Return:
      `True` if the tensor contains any NaN element, or `False` otherwise.
    """
    logger = log.get_logger(name)
    if logger.isEnabledFor(log.DEBUG) and tensor.dtype in _TENSOR_REAL:
        num_nan = torch.isnan(tensor).sum().item()
        if num_nan > 0:
            per_nan = num_nan / tensor.numel()
            msg = (
                "{:d} ({:.2%}) NaN values found".format(num_nan, per_nan)
                if msg is None
                else msg.format(abs_num=num_nan, rel_num=per_nan, **kwargs)
            )
            if raise_exception:
                raise ValueError(msg)
            else:
                logger.debug(msg)
            return True
    return False
