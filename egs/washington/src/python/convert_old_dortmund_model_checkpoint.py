from __future__ import absolute_import

import argparse
import os
import re
from collections import OrderedDict

import torch


def convert_old_parameters(params):
    """Convert parameters from the old model to the new one."""
    # type: OrderedDict -> OrderedDict
    new_params = []
    for k, v in params.items():
        if k[:4] == "conv":
            k = "conv." + k
        elif k[:2] == "fc":
            k = "fc." + k
        new_params.append((k, v))
    return OrderedDict(new_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint", help="File path of the input checkpoint")
    parser.add_argument("output_checkpoint", help="File path of the output checkpoint")
    args = parser.parse_args()
    # Check input checkpoint
    assert os.path.isfile(args.input_checkpoint), "{!r} is not a file".format(
        args.input_checkpoint
    )
    # Prepare directory for the output checkpoint
    outdir = os.path.dirname(args.output_checkpoint)
    if os.path.exists(outdir):
        assert os.path.isdir(outdir), "{!r} is not a directory".format(outdir)
    else:
        os.makedirs(outdir)

    params = torch.load(args.input_checkpoint)
    params = convert_old_parameters(params)
    torch.save(params, args.output_checkpoint)
