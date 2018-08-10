#!/usr/bin/env python

from __future__ import print_function

import io
import os
import re
import subprocess
import tempfile

from typing import AnyStr, Optional, Dict

from laia.utils.symbols_table import SymbolsTable

DEV_NULL = io.open("/dev/null", "w")


def get_include_words(queries, syms):
    # type: (Optional[AnyStr], SymbolsTable) -> str
    if queries:
        include_words = []
        with io.open(queries, "r") as f:
            for line in f:
                word_id = syms[line.strip()]
                if word_id is not None:
                    include_words.append(str(word_id))
        return " ".join(include_words)
    else:
        return ""


def get_include_words_set(queries):
    if queries:
        with io.open(queries, "r") as f:
            return {line.strip() for line in f}
    else:
        return {}


def make_index_utterance_process(
    syms,  # type: SymbolsTable
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    # type (...) -> subprocess.Popen
    return subprocess.Popen(
        [
            "lattice-word-index-utterance",
            "--print-args=false",
            "--include-words=%s" % get_include_words(queries, syms),
            "--acoustic-scale=%f" % acoustic_scale,
            "--num-threads=4",
            "--verbose=1",
            "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )


def make_index_segment_process(
    syms,  # type: SymbolsTable
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    # type (...) -> subprocess.Popen
    return subprocess.Popen(
        [
            "lattice-word-index-segment",
            "--print-args=false",
            "--include-words=%s" % get_include_words(queries, syms),
            "--acoustic-scale=%f" % acoustic_scale,
            "--num-threads=4",
            "--verbose=1",
            "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )


def make_index_position_process(
    syms,  # type: SymbolsTable
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    # type (...) -> subprocess.Popen
    return subprocess.Popen(
        [
            "lattice-word-index-position",
            "--print-args=false",
            "--include-words=%s" % get_include_words(queries, syms),
            "--acoustic-scale=%f" % acoustic_scale,
            "--num-threads=4",
            "--verbose=1",
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
    return subprocess.Popen(
        [
            "lattice-to-word-frame-post",
            "--print-args=false",
            "--acoustic-scale=%f" % acoustic_scale,
            "--verbose=1",
            "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )


def make_simple_kws_eval_process(kws_ref, queries=None):
    # type (AnyStr, Optional[AnyStr]) -> subprocess.Popen
    return subprocess.Popen(
        [
            "SimpleKwsEval",
            "--query_set",
            queries if queries else "",
            "--interpolated_precision",
            "true",
            "--collapse_matches",
            "true",
            "--sort",
            "desc",
            kws_ref,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )


def make_kws_assessment_process(table_file, queries=None):
    # type: (Optional[AnyStr]) -> subprocess.Popen
    popen_args = (
        ["kws-assessment-joan", "-a", "-m", "-t"]
        + (["-w", queries] if queries else [])
        + [table_file]
    )
    return subprocess.Popen(popen_args, stdout=subprocess.PIPE)


def simple_kws_eval_utterance_index(
    syms,  # type: SymbolsTable
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_utterance_process(
        syms, lattice_ark, acoustic_scale, queries, verbose
    )
    p2 = make_simple_kws_eval_process(kws_ref, queries)
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        for i in range(1, len(line), 3):
            word = syms[int(line[i])]
            score = line[i + 1]
            p2.stdin.write("{} {} {}\n".format(word, utt, score))
    p1.stdout.close()
    out = p2.communicate()[0]
    return out.splitlines()


def simple_kws_eval_segment_index(
    syms,  # type: SymbolsTable
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_utterance_process(
        syms, lattice_ark, acoustic_scale, queries, verbose
    )
    p2 = make_simple_kws_eval_process(kws_ref, queries)
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        seen_words = set()
        for i in range(1, len(line), 5):
            word = syms[int(line[i])]
            score = line[i + 3]
            if word not in seen_words:
                p2.stdin.write("{} {} {}\n".format(word, utt, score))
                seen_words.add(word)
    p1.stdout.close()
    out = p2.communicate()[0]
    return out.splitlines()


def get_kws_ref_set(kws_ref):
    kws_ref_set = set()
    with io.open(kws_ref, "r") as f:
        for line in f:
            kws_ref_set.add(tuple(line.split()))
    return kws_ref_set


def add_missing_words(kws_ref_set, kws_hyp_set, tmpf):
    for word, utt in kws_ref_set:
        if (word, utt) not in kws_hyp_set:
            tmpf.write("{} {} {} {}\n".format(utt, word, 1, "-inf"))
    tmpf.close()


def kws_assessment_parse_output(out):
    # type: (AnyStr) -> Dict[AnyStr, float]
    mAP = float(re.search(r" MAP = ([0-9.]+)", out, re.MULTILINE).group(1))
    gAP = float(re.search(r" AP = ([0-9.]+)", out, re.MULTILINE).group(1))
    return {"mAP": mAP, "gAP": gAP}


def kws_assessment_utterance_index(
    syms,  # type: SymbolsTable
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_utterance_process(
        syms, lattice_ark, acoustic_scale, queries, verbose
    )

    kws_ref_set = get_kws_ref_set(kws_ref)
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        for i in range(1, len(line), 3):
            word = syms[int(line[i])]
            score = line[i + 1]
            rel = 1 if (word, utt) in kws_ref_set else 0
            tmpf.write("{} {} {} {}\n".format(utt, word, rel, score))
            kws_hyp_set.add((word, utt))
    p1.stdout.close()
    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)

    p2 = make_kws_assessment_process(tmppath, queries)
    out = p2.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


def kws_assessment_segment_index(
    syms,  # type: SymbolsTable
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_segment_process(syms, lattice_ark, acoustic_scale, queries, verbose)

    kws_ref_set = get_kws_ref_set(kws_ref)
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        seen_words = set()
        for i in range(1, len(line), 5):
            word = syms[int(line[i])]
            score = line[i + 3]
            rel = 1 if (word, utt) in kws_ref_set else 0
            if word not in seen_words:
                tmpf.write("{} {} {} {}\n".format(utt, word, rel, score))
                kws_hyp_set.add((word, utt))
                seen_words.add(word)
    p1.stdout.close()

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p2 = make_kws_assessment_process(tmppath, queries)
    out = p2.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


def kws_assessment_position_index(
    syms,  # type: SymbolsTable
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_index_position_process(
        syms, lattice_ark, acoustic_scale, queries, verbose
    )

    kws_ref_set = get_kws_ref_set(kws_ref)
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        seen_words = set()
        for i in range(1, len(line), 6):
            word = syms[int(line[i])]
            score = line[i + 4]
            rel = 1 if (word, utt) in kws_ref_set else 0
            if word not in seen_words:
                tmpf.write("{} {} {} {}\n".format(utt, word, rel, score))
                kws_hyp_set.add((word, utt))
                seen_words.add(word)
    p1.stdout.close()

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p2 = make_kws_assessment_process(tmppath, queries)
    out = p2.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


def kws_assessment_column_index(
    syms,  # type: SymbolsTable
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
):
    p1 = make_posteriorgram_process(lattice_ark, acoustic_scale, verbose)

    queries_set = get_include_words_set(queries)
    kws_ref_set = get_kws_ref_set(kws_ref)
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.strip()
        m = re.match(r"^([^ ]+) (.)+$", line)
        utt = m.group(1)
        best_score = {}
        for m in re.finditer(r"\[ ([0-9]+ [0-9.e-]+ )+\]", line):
            frame = m.group(0)
            for m in re.finditer(r"([0-9]+) ([0-9.e-]+) ", frame):
                word = syms[int(m.group(1))]
                score = float(m.group(2))
                if word in queries_set and (
                    word not in best_score or best_score[word] < score
                ):
                    best_score[word] = score

        for word, score in best_score.items():
            rel = 1 if (word, utt) in kws_ref_set else 0
            tmpf.write("{} {} {} {}\n".format(utt, word, rel, score))
            kws_hyp_set.add((word, utt))
    p1.stdout.close()

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p2 = make_kws_assessment_process(tmppath, queries)
    out = p2.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-scale", default=1.0, type=float)
    parser.add_argument("--queries", type=str, default=None)
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
    args = parser.parse_args()

    if args.index_type == "utterance":
        if args.use_kws_eval:
            func = simple_kws_eval_utterance_index
        else:
            func = kws_assessment_utterance_index
    elif args.index_type == "segment":
        if args.use_kws_eval:
            func = simple_kws_eval_segment_index
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
            func = kws_assessment_column_index
    else:
        raise NotImplementedError

    syms = SymbolsTable(args.syms)
    print(
        func(
            syms,
            args.kws_refs,
            args.lattice_ark,
            args.acoustic_scale,
            args.queries,
            args.verbose,
        )
    )
