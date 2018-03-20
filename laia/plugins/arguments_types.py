from __future__ import absolute_import

import argparse
import logging
from collections import OrderedDict


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def str2num_accept_closed_range(v, t, vmin=None, vmax=None):
    v = t(v)
    if vmin is not None and v < vmin:
        raise argparse.ArgumentTypeError(
            'Value must be lower than {}'.format(vmin))
    if vmax is not None and v > vmax:
        raise argparse.ArgumentTypeError(
            'Value must be greater than {}'.format(vmax))
    return v


def str2num_accept_open_range(v, t, vmin=None, vmax=None):
    v = t(v)
    if vmin is not None and v <= vmin:
        raise argparse.ArgumentTypeError(
            'Value must be lower than {}'.format(vmin))
    if vmax is not None and v >= vmax:
        raise argparse.ArgumentTypeError(
            'Value must be greater than {}'.format(vmax))
    return v


def str2loglevel(v):
    vmap = OrderedDict([
        ('debug', logging.DEBUG),
        ('info', logging.INFO),
        ('warning', logging.WARNING),
        ('error', logging.ERROR),
        ('critical', logging.CRITICAL)
    ])
    try:
        return vmap[v.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('Valid logging levels are: {!r}',
                                         vmap.values())


class NumberInClosedRange(object):
    def __init__(self, type, vmin=None, vmax=None):
        self._type = type
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, v):
        return str2num_accept_closed_range(
            v, self._type, self._vmin, self._vmax)


class NumberInOpenRange(object):
    def __init__(self, type, vmin=None, vmax=None):
        self._type = type
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, v):
        return str2num_accept_open_range(
            v, self._type, self._vmin, self._vmax)
