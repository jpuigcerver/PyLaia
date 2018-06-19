from __future__ import absolute_import

import argparse
import os
import re
import torch
from collections import OrderedDict


def convert_old_parameters(params):
    """Convert parameters from the old model to the new one."""
    # type: OrderedDict -> OrderedDict
    new_params = []
    for k, v in params.items():
        m = re.match(r"^conv_block([0-9]+)\.([a-z_.]+)$", k)
        if m:
            if m.group(2) == "poolsize":
                pass
            else:
                new_params.append(("conv.{}.{}".format(m.group(1), m.group(2)), v))
        elif k[0] == "_":
            pass
        else:
            new_params.append((k, v))
    return OrderedDict(new_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint", help="File path of the input checkpoint")
    parser.add_argument("output_checkpoint", help="File path of the output checkpoint")
    args = parser.parse_args()
    assert os.path.isfile(args.input_checkpoint)

    params = torch.load(args.input_checkpoint)
    params = convert_old_parameters(params)
    torch.save(params, args.output_checkpoint)
