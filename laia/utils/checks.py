from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import torch


_TENSOR_REAL = [torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor]
if torch.cuda.is_available():
    _TENSOR_REAL += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                     torch.cuda.HalfTensor]
_TENSOR_REAL = tuple(_TENSOR_REAL)


def check_inf(tensor, msg=None, logger=None, raise_exception=False, **kwargs):
    r"""Check whether a tensor contains a +/- infinite value.

    Arguments:
      tensor (torch.Tensor): tensor to check.
      msg (str): message format string. The message format can use the keys
          ``abs_num`` and ``rel_num`` to print the absolute number and the
           percentage of infinite elements. (Default: None)
      logger (logging.Logger): logger used to log the event (Default: None)
      raise_exception (bool): raise an exception instead of logging the event
          (Default: False)
      kwargs: additional named arguments passed to format the message.

    Return:
      `True` if the tensor contains any +/- infinite element, or `False`
      otherwise.
    """
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data

    if logger is None:
        logger = logging.getLogger()

    if isinstance(tensor, _TENSOR_REAL) and logger.isEnabledFor(logging.DEBUG):
        inf = float(np.inf)
        num_inf = torch.sum(tensor == inf) + torch.sum(tensor == -inf)
        if num_inf > 0:
            per_inf = num_inf / tensor.numel()
            if msg is None:
                msg = '{:d} ({:.2%}) INF values found'.format(num_inf, per_inf)
            else:
                msg = msg.format(abs_num=num_inf, rel_num=per_inf, **kwargs)

            if raise_exception:
                raise ValueError(msg)
            else:
                logger.debug(msg)

            return True

    # No +/- inf were found
    return False


def check_nan(tensor, msg=None, logger=None, raise_exception=False, **kwargs):
    r"""Check whether a tensor contains a NaN value.

    Arguments:
      tensor (torch.Tensor): tensor to check.
      msg (str): message format string. The message format can use the keys
          ``abs_num`` and ``rel_num`` to print the absolute number and the
           percentage of NaN elements. (Default: None)
      logger (logging.Logger): logger used to log the event (Default: None)
      raise_exception (bool): raise an exception instead of logging the event
          (Default: False)
      kwargs: additional named arguments passed to format the message.

    Return:
      `True` if the tensor contains any NaN element, or `False` otherwise.
    """
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data

    if logger is None:
        logger = logging.getLogger()

    if isinstance(tensor, _TENSOR_REAL) and logger.isEnabledFor(logging.DEBUG):
        num_nan = torch.sum(tensor != tensor)
        if num_nan > 0:
            per_nan = num_nan / tensor.numel()
            if msg is None:
                msg = '{:d} ({:.2%}) INF values found'.format(num_nan, per_nan)
            else:
                msg = msg.format(abs_num=num_nan, rel_num=per_nan, **kwargs)

            if raise_exception:
                raise ValueError(msg)
            else:
                logger.debug(msg)

            return True

    # No NaNs were found
    return False
