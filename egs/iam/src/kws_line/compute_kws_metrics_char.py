#!/usr/bin/env python

from __future__ import print_function

import io
import os
import subprocess
import tempfile

from typing import AnyStr, Optional, Iterable

from compute_kws_metrics_word import (
    add_missing_words,
    get_include_words_set,
    get_kws_ref_set,
    make_kws_assessment_process,
    kws_assessment_parse_output,
)
from laia.utils.symbols_table import SymbolsTable

DEV_NULL = io.open("/dev/null", "w")


def make_index_segment_process(
    delimiters,  # type: Iterable[int]
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    verbose=False,  # type: bool
):
    # type (...) -> subprocess.Popen
    return subprocess.Popen(
        [
            "lattice-char-index-segment",
            "--print-args=false",
            "--acoustic-scale=%f" % acoustic_scale,
            "--nbest=%d" % nbest,
            "--num-threads=4",
            "--verbose=1",
            " ".join([str(s) for s in delimiters]),
            "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )


def make_index_position_process(
    delimiters,  # type: Iterable[int]
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    verbose=False,  # type: bool
):
    # type (...) -> subprocess.Popen
    return subprocess.Popen(
        [
            "lattice-char-index-position",
            "--print-args=false",
            "--acoustic-scale=%f" % acoustic_scale,
            "--nbest=%d" % nbest,
            "--num-threads=4",
            "--verbose=1",
            " ".join([str(s) for s in delimiters]),
            "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )


def make_posteriorgram_process(
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    verbose=False,  # type: bool
):
    p1 = subprocess.Popen(
        [
            "lattice-expand-subpaths",
            "--print-args=false",
            "--num-threads=4",
            "--symbol-table=",
            "--symbol-table-text=true" "ark:%s" % lattice_ark,
            "ark:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )
    p2 = subprocess.Popen(
        [
            "lattice-to-word-frame-post",
            "--print-args=false",
            "--acoustic-scale=%f" % acoustic_scale,
            "--verbose=1",
            "ark:-",
            "ark,t:-",
        ],
        stdin=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )
    return p1, p2


def kws_assessment_segment_index(
    syms,  # type: SymbolsTable
    delimiters,  # type: Iterable[int]
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_segment_process(
        delimiters, lattice_ark, acoustic_scale, nbest, verbose
    )

    kws_ref_set = get_kws_ref_set(kws_ref)
    queries_set = get_include_words_set(queries) if queries else None
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        seen_words = set()
        for i in range(1, len(line), 5):
            word = "".join([syms[int(x)] for x in line[i].split("_")])
            score = line[i + 3]
            rel = 1 if (word, utt) in kws_ref_set else 0
            if (queries_set is None or word in queries_set) and word not in seen_words:
                tmpf.write("{} {} {} {}\n".format(utt, word, rel, score))
                kws_hyp_set.add((word, utt))
                seen_words.add(word)
    p1.stdout.close()

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p2 = make_kws_assessment_process(tmppath, queries, verbose)
    out = p2.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


def kws_assessment_position_index(
    syms,  # type: SymbolsTable
    delimiters,  # type: Iterable[int]
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_position_process(
        delimiters, lattice_ark, acoustic_scale, nbest, verbose
    )

    kws_ref_set = get_kws_ref_set(kws_ref)
    queries_set = get_include_words_set(queries) if queries else None
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        seen_words = set()
        for i in range(1, len(line), 6):
            word = "".join([syms[int(x)] for x in line[i].split("_")])
            score = line[i + 4]
            rel = 1 if (word, utt) in kws_ref_set else 0
            if (queries_set is None or word in queries_set) and word not in seen_words:
                tmpf.write("{} {} {} {}\n".format(utt, word, rel, score))
                print("{} {} {} {}\n".format(utt, word, rel, score))
                kws_hyp_set.add((word, utt))
                seen_words.add(word)
    p1.stdout.close()

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p2 = make_kws_assessment_process(tmppath, queries, verbose)
    out = p2.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-scale", default=1.0, type=float)
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--nbest", type=int, default=100)
    parser.add_argument(
        "--index-type",
        choices=("utterance", "segment", "position", "column"),
        default="utterance",
    )
    parser.add_argument("--use-kws-eval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("syms")
    parser.add_argument("kws_refs")
    parser.add_argument("lattice_ark")
    parser.add_argument("delimiters", type=int, nargs="+")
    args = parser.parse_args()

    if args.index_type == "utterance":
        if args.use_kws_eval:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.index_type == "segment":
        if args.use_kws_eval:
            raise NotImplementedError
        else:
            func = kws_assessment_segment_index
    elif args.index_type == "position":
        if args.use_kws_eval:
            raise NotImplementedError
        else:
            func = kws_assessment_position_index
    elif args.index_type == "column":
        if args.use_kws_eval:
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    syms = SymbolsTable(args.syms)
    print(
        func(
            syms,
            args.delimiters,
            args.kws_refs,
            args.lattice_ark,
            args.acoustic_scale,
            args.nbest,
            args.queries,
            args.verbose,
        )
    )
