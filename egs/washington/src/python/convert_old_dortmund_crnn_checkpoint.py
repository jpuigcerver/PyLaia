#!/usr/bin/env python3
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
        elif k[:6] == "linear":
            k = k.split(".")
            assert len(k) == 3
            k = k[0] + "." + k[2]
        new_params.append((k, v))
    return OrderedDict(new_params)

['conv1_1.weight', 'conv1_1.bias', 'conv1_2.weight', 'conv1_2.bias', 'conv2_1.weight', 'conv2_1.bias', 'conv2_2.weight', 'conv2_2.bias', 'conv3_1.weight', 'conv3_1.bias', 'conv3_2.weight', 'conv3_2.bias', 'conv3_3.weight', 'conv3_3.bias', 'conv3_4.weight', 'conv3_4.bias', 'conv3_5.weight', 'conv3_5.bias', 'conv3_6.weight', 'conv3_6.bias', 'conv4_1.weight', 'conv4_1.bias', 'conv4_2.weight', 'conv4_2.bias', 'conv4_3.weight', 'conv4_3.bias', 'blstm.weight_ih_l0', 'blstm.weight_hh_l0', 'blstm.bias_ih_l0', 'blstm.bias_hh_l0', 'blstm.weight_ih_l0_reverse', 'blstm.weight_hh_l0_reverse', 'blstm.bias_ih_l0_reverse', 'blstm.bias_hh_l0_reverse', 'linear._module.weight', 'linear._module.bias']



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
    elif outdir:
        os.makedirs(outdir)

    params = torch.load(args.input_checkpoint)
    params = convert_old_parameters(params)
    torch.save(params, args.output_checkpoint)
