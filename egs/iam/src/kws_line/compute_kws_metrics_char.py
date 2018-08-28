#!/usr/bin/env python

from __future__ import print_function

import io
import os
import re
import subprocess
import tempfile

from typing import AnyStr, Optional, Iterable

from compute_kws_metrics_word import (
    add_missing_words,
    get_include_words,
    get_include_words_set,
    get_kws_ref_set,
    make_kws_assessment_process,
    kws_assessment_parse_output,
)
from laia.utils.symbols_table import SymbolsTable

DEV_NULL = io.open("/dev/null", "w")


def get_tmp_filename():
    fd, tmppath = tempfile.mkstemp()
    os.close(fd)
    os.remove(tmppath)
    return tmppath


def make_prune_lattice(lattice_ark, max_states=None, max_arcs=None, verbose=False):
    if max_states or max_arcs:
        args = ["lattice-prune-dyn-beam"]
        if max_states:
            args.append("--max-states=%d" % max_states)
        if max_arcs:
            args.append("--max-arcs=%d" % max_arcs)

        args.append("ark:%s" % lattice_ark)
        args.append("ark:-")
        p = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=None if verbose else DEV_NULL
        )
        return p
    else:
        return None


def make_index_segment_process(
    delimiters,  # type: Iterable[int]
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    # type (...) -> (subprocess.Popen)
    p1 = make_prune_lattice(lattice_ark, max_states, max_arcs, verbose)
    p2 = subprocess.Popen(
        [
            "lattice-char-index-segment",
            "--print-args=false",
            "--acoustic-scale=%f" % acoustic_scale,
            "--nbest=%d" % nbest,
            "--num-threads=4",
            "--verbose=1",
            " ".join([str(s) for s in delimiters]),
            "ark:-" if p1 else "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdin=p1.stdout if p1 else None,
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )
    if p1:
        p1.stdout.close()
    return p2


def make_index_position_process(
    delimiters,  # type: Iterable[int]
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    # type (...) -> (subprocess.Popen)
    p1 = make_prune_lattice(lattice_ark, max_states, max_arcs, verbose)
    p2 = subprocess.Popen(
        [
            "lattice-char-index-position",
            "--print-args=false",
            "--acoustic-scale=%f" % acoustic_scale,
            "--nbest=%d" % nbest,
            "--num-threads=4",
            "--verbose=1",
            " ".join([str(s) for s in delimiters]),
            "ark:-" if p1 else "ark:%s" % lattice_ark,
            "ark,t:-",
        ],
        stdin=p1.stdout if p1 else None,
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )
    if p1:
        p1.stdout.close()
    return p2


def make_posteriorgram_process(
    delimiters,  # type: Iterable[int]
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    # type (...) -> (subprocess.Popen, str)
    word_symbols_table = get_tmp_filename()

    p1 = make_prune_lattice(lattice_ark, max_states, max_arcs, verbose)
    p2 = subprocess.Popen(
        [
            "lattice-expand-subpaths",
            "--print-args=true",
            "--num-threads=4",
            "--symbol-table=%s" % word_symbols_table,
            "--symbol-table-text=true",
            " ".join([str(x) for x in delimiters]),
            "ark:-" if p1 else "ark:%s" % lattice_ark,
            "ark:-",
        ],
        stdin=p1.stdout if p1 else None,
        stderr=None if verbose else DEV_NULL,
        stdout=subprocess.PIPE,
    )
    if p1:
        p1.stdout.close()
    p3 = subprocess.Popen(
        [
            "lattice-to-word-frame-post",
            "--print-args=true",
            "--acoustic-scale=%f" % acoustic_scale,
            "--verbose=1",
            "ark:-",
            "ark,t:-",
        ],
        stdin=p2.stdout,
        stderr=None if verbose else DEV_NULL,
        stdout=subprocess.PIPE,
    )
    return p3, word_symbols_table


def make_index_utterance_process(
    delimiters,  # type: Iterable[int]
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    syms,  # type: SymbolsTable
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
    queries=None,  # type: Optional[AnyStr]
):
    # type (...) -> (subprocess.Popen, str)
    word_symbols_table = get_tmp_filename()
    expanded_lattice = get_tmp_filename()

    p1 = make_prune_lattice(lattice_ark, max_states, max_arcs, verbose)
    p2 = subprocess.Popen(
        [
            "lattice-expand-subpaths",
            "--print-args=false",
            "--num-threads=4",
            "--symbol-table=%s" % word_symbols_table,
            "--symbol-table-text=true",
            " ".join([str(x) for x in delimiters]),
            "ark:-" if p1 else "ark:%s" % lattice_ark,
            "ark:%s" % expanded_lattice,
        ],
        stdin=p1.stdout if p1 else None,
        stderr=None if verbose else DEV_NULL,
    )
    if p1:
        p1.stdout.close()
    p2.communicate()

    if queries:
        queries_set = get_include_words_set(queries)

        include_words = set()
        with io.open(word_symbols_table, "r", encoding="utf-8") as f:
            for line in f:
                word_chars, word_id = line.split()
                word = "".join([syms[int(x)] for x in word_chars.split("_")])
                if word in queries_set:
                    include_words.add(word_id)

        include_words = " ".join(include_words)
    else:
        include_words = ""

    p3 = subprocess.Popen(
        [
            "lattice-word-index-utterance",
            "--print-args=false",
            "--include-words=%s" % include_words,
            "--acoustic-scale=%f" % acoustic_scale,
            "--num-threads=4",
            "--verbose=1",
            "ark:%s" % expanded_lattice,
            "ark,t:-",
        ],
        stderr=None if verbose else DEV_NULL,
        stdout=subprocess.PIPE,
    )
    return p3, word_symbols_table


def kws_assessment_segment_index(
    syms,  # type: SymbolsTable
    delimiters,  # type: Iterable[int]
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    p1 = make_index_segment_process(
        delimiters, lattice_ark, acoustic_scale, nbest, verbose, max_states, max_arcs
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
    p3 = make_kws_assessment_process(tmppath, queries, verbose)
    out = p3.communicate()[0]
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
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    p1 = make_index_position_process(
        delimiters, lattice_ark, acoustic_scale, nbest, verbose, max_states, max_arcs
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
                kws_hyp_set.add((word, utt))
                seen_words.add(word)
    p1.stdout.close()

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p3 = make_kws_assessment_process(tmppath, queries, verbose)
    out = p3.communicate()[0]
    os.remove(tmppath)
    return kws_assessment_parse_output(out)


def kws_assessment_column_index(
    syms,  # type: SymbolsTable
    delimiters,  # type: Iterable[int]
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    p1, word_syms_str = make_posteriorgram_process(
        delimiters, lattice_ark, acoustic_scale, verbose, max_states, max_arcs
    )

    index = {}
    for line in p1.stdout:
        line = line.strip()
        m = re.match(r"^([^ ]+) (.)+$", line)
        utt = m.group(1)
        best_score = {}
        for m in re.finditer(r"\[ ([0-9]+ [0-9.e-]+ )+\]", line):
            frame = m.group(0)
            for m in re.finditer(r"([0-9]+) ([0-9.e-]+) ", frame):
                word = int(m.group(1))
                score = float(m.group(2))
                if word not in best_score or best_score[word] < score:
                    best_score[word] = score
        aux = [(score, word) for word, score in best_score.items()]
        aux.sort(reverse=True)
        index[utt] = aux[:nbest]

    word_syms = SymbolsTable(word_syms_str)
    queries_set = get_include_words_set(queries)
    kws_ref_set = get_kws_ref_set(kws_ref)
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for utt in index:
        for score, word in index[utt]:
            word = "".join([syms[int(x)] for x in word_syms[word].split("_")])
            if queries_set is None or word in queries_set:
                rel = 1 if (word, utt) in kws_ref_set else 0
                tmpf.write(u"{} {} {} {}\n".format(utt, word, rel, score))
                kws_hyp_set.add((word, utt))

    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)
    p2 = make_kws_assessment_process(tmppath, queries, verbose)
    out = p2.communicate()[0]
    os.remove(tmppath)
    os.remove(word_syms_str)
    return kws_assessment_parse_output(out)


def kws_assessment_utterance_index(
    syms,  # type: SymbolsTable
    delimiters,  # type: Iterable[int]
    kws_ref,  # type: AnyStr
    lattice_ark,  # type: AnyStr
    acoustic_scale,  # type: float
    nbest,  # type: int
    queries=None,  # type: Optional[AnyStr]
    verbose=False,  # type: bool
    max_states=None,  # type: Optional[int]
    max_arcs=None,  # type: Optional[int]
):
    p1, word_syms_str = make_index_utterance_process(
        delimiters,
        lattice_ark,
        acoustic_scale,
        syms,
        verbose,
        max_states,
        max_arcs,
        queries,
    )

    word_syms = SymbolsTable(word_syms_str)
    kws_ref_set = get_kws_ref_set(kws_ref)
    fd, tmppath = tempfile.mkstemp()
    tmpf = os.fdopen(fd, "w")
    kws_hyp_set = set()
    for line in p1.stdout:
        line = line.split()
        utt = line[0]
        for i in range(1, len(line), 3):
            word = word_syms[int(line[i])]
            word = "".join([syms[int(x)] for x in word.split("_")])
            score = line[i + 1]
            rel = 1 if (word, utt) in kws_ref_set else 0
            tmpf.write(u"{} {} {} {}\n".format(utt, word, rel, score))
            kws_hyp_set.add((word, utt))
    p1.stdout.close()
    add_missing_words(kws_ref_set, kws_hyp_set, tmpf)

    p2 = make_kws_assessment_process(tmppath, queries, verbose)
    out = p2.communicate()[0]
    os.remove(tmppath)
    os.remove(word_syms_str)
    return kws_assessment_parse_output(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-scale", default=1.0, type=float)
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--nbest", type=int, default=100)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--max-arcs", type=int, default=None)
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
            func = kws_assessment_utterance_index
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
            func = kws_assessment_column_index
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
            args.max_states,
            args.max_arcs,
        )
    )
