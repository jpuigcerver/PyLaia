#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import io
import numpy as np
import logging

import hyperopt


def load_data(fname):
    f = io.open(fname, "r", encoding="utf-8")
    voc_counts = {}
    for line in f:
        for word in line.split():
            if word in voc_counts:
                voc_counts[word] += 1
            else:
                voc_counts[word] = 1
    f.close()
    return voc_counts


def normalize_weights(weights):
    weights = np.append(np.asarray(weights, dtype=np.float32), [0.0])
    weights = np.exp(weights - weights.max())
    weights = weights / weights.sum()
    return weights


def compute_voc_from_weights(tr_cnts, weights, voc_size):
    weight_cnt = {}
    for w, tr_cnt in zip(weights, tr_cnts):
        for word, cnt in tr_cnt.items():
            weight_cnt[word] = weight_cnt.get(word, 0) + w * cnt
    weight_cnt = list([(cnt, word) for word, cnt in weight_cnt.items()])
    weight_cnt.sort(reverse=True)
    weight_cnt = weight_cnt[:voc_size]
    return [word for _, word in weight_cnt]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0x1234,
                        help="Seed for the PRNG")
    parser.add_argument("--max_iters", type=int, default=100,
                        help="Max iterations")
    parser.add_argument("--verbose", action="store_true",
                        help="If true, shows the result of each trial")
    parser.add_argument("voc_size", type=int, help="Size of the vocabulary")
    parser.add_argument("train_txt", nargs="+", help="Training data")
    parser.add_argument("valid_txt", help="Validation data")
    args = parser.parse_args()

    va_cnt = load_data(args.valid_txt)
    tr_cnts = [load_data(x) for x in args.train_txt]
    N = len(tr_cnts)
    NWORDS = sum([cnt for word, cnt in va_cnt.items()])

    logging.basicConfig()
    logger = logging.getLogger('create_vocabulary.py')
    if args.verbose:
        logger.setLevel(logging.INFO)

    def compute_running_oov(weights):
        # Normalize weights (softmax)
        weights = normalize_weights(weights)
        # Compute weighted vocabulary, limited to voc_size
        voc = set(compute_voc_from_weights(tr_cnts, weights, args.voc_size))
        # Compute running OOV
        oov = sum([cnt for word, cnt in va_cnt.items() if word not in voc])
        oov = oov / NWORDS
        logger.info(" ".join(["%f" % x for x in weights] + [str(oov)]))
        return oov


    space = [hyperopt.hp.uniform("w%d" % i, -10, 10) for i in range(N - 1)]
    best = hyperopt.fmin(
        fn=compute_running_oov,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=args.max_iters,
        rstate=np.random.RandomState(args.seed),
    )

    wopt = [best["w%d" % i] for i in range(N - 1)]
    wopt = normalize_weights(wopt)
    for word in compute_voc_from_weights(tr_cnts, wopt, args.voc_size):
        print(word)
