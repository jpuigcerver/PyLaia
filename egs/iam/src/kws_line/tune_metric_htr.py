#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys
from functools import lru_cache
from shlex import quote
import numpy as np

from compute_metric_htr import htr_assessment
from hyperopt import fmin, tpe, hp

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("tune_kws_metric")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-scale-max", default=5.0, type=float)
    parser.add_argument("--acoustic-scale-min", default=0.1, type=float)
    parser.add_argument("--acoustic-scale-quant", default=0.05, type=float)
    parser.add_argument("--prior-scale-max", default=1.0, type=float)
    parser.add_argument("--prior-scale-min", default=0.0, type=float)
    parser.add_argument("--prior-scale-quant", default=0.1, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wspace", default="<space>")
    parser.add_argument("--max-iters", type=int, default=400)
    parser.add_argument("--optimize-for", choices=("CER", "WER", "avg"), default="avg")
    parser.add_argument("--seed", type=int, default=0x12345)
    parser.add_argument("--char-separator", default="")
    parser.add_argument("syms")
    parser.add_argument("model")
    parser.add_argument("lattice_ark_pattern")
    parser.add_argument("ref_txt")
    args = parser.parse_args()
    logger.info("Command line: %s", " ".join([quote(x) for x in sys.argv[1:]]))

    # Configure hyperparamter search space
    space = []
    params_to_optimize = []
    if args.prior_scale_max != args.prior_scale_min:
        params_to_optimize.append("prior_scale")
        space.append(
            hp.quniform(
                "prior_scale",
                args.prior_scale_min,
                args.prior_scale_max,
                args.prior_scale_quant,
            )
        )
        prior_scale_global = None
        prior_scale_key = len(space) - 1
    else:
        prior_scale_global = args.prior_scale_max
        prior_scale_key = None

    if args.acoustic_scale_max != args.acoustic_scale_min:
        params_to_optimize.append("acoustic_scale")
        space.append(
            hp.quniform(
                "acoustic_scale",
                args.acoustic_scale_min,
                args.acoustic_scale_max,
                args.acoustic_scale_quant,
            )
        )
        acoustic_scale_global = None
        acoustic_scale_key = len(space) - 1
    else:
        acoustic_scale_global = args.acoustic_scale_max
        acoustic_scale_key = None

    @lru_cache(maxsize=None)
    def objective(params):
        if prior_scale_key is not None:
            prior_scale = params[prior_scale_key]
        else:
            prior_scale = prior_scale_global

        if acoustic_scale_key is not None:
            acoustic_scale = params[acoustic_scale_key]
        else:
            acoustic_scale = acoustic_scale_global

        lattice_ark = args.lattice_ark_pattern.format(
            acoustic_scale=acoustic_scale, prior_scale=prior_scale
        )
        logging.debug(
            "Trying acoustic_scale={:.2f} with lattic_ark={}".format(
                acoustic_scale, lattice_ark
            )
        )
        result = htr_assessment(
            ref_txt=args.ref_txt,
            lattice_ark=lattice_ark,
            model=args.model,
            syms=args.syms,
            wspace=args.wspace,
            acoustic_scale=acoustic_scale,
            char_sep=args.char_separator,
            verbose=args.verbose,
        )
        logger.info(
            "acoustic_scale = {:.2f}  prior_scale = {:.1f}  "
            "CER = {:.2f}  WER = {:.2f}".format(
                acoustic_scale, prior_scale, result["CER"], result["WER"]
            )
        )
        if args.optimize_for == "CER":
            return result["CER"]
        elif args.optimize_for == "WER":
            return result["WER"]
        else:
            return (result["CER"] + result["WER"]) / 2.0

    logger.info(
        "Optimizing {} for: {}".format(str(params_to_optimize), args.optimize_for)
    )
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_iters,
        rstate=np.random.RandomState(args.seed),
    )
    print(best)
