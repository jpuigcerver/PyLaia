#!/usr/bin/env python3

from __future__ import print_function

import io
import os
import re
import subprocess
import tempfile

DEV_NULL = io.open("/dev/null", "w")


def get_tmp_filename():
    fd, tmppath = tempfile.mkstemp()
    os.close(fd)
    os.remove(tmppath)
    return tmppath


def get_word_from_char_txt(txt, wspace, char_sep=""):
    out = tempfile.NamedTemporaryFile()
    p = subprocess.Popen(
        [
            "awk",
            """
            {
              printf("%%s ", $1);
              for (i = 2; i <= NF; ++i) {
                if ($i == "%s") { printf(" "); }
                else { printf("%s%%s", $i); }
              }
              printf("\\n");
            }
            """
            % (wspace, char_sep),
            txt if isinstance(txt, str) else txt.name,
        ],
        stdout=out,
    )
    p.communicate()

    return out


def compute_err(ref_txt, hyp_txt):
    p = subprocess.Popen(
        ["compute-wer", "--print-args=false", "ark:%s" % ref_txt, "ark:%s" % hyp_txt],
        stdout=subprocess.PIPE,
    )
    out = p.communicate()[0].decode("utf-8")
    m = re.search(r"%WER ([0-9.]+)", out, re.MULTILINE)
    assert m is not None
    return float(m.group(1))


def htr_assessment(ref_txt, lattice_ark, model, syms, wspace, acoustic_scale, char_sep, verbose):
    # Convert to character lattice
    p1 = subprocess.Popen(
        [
            "lattice-to-phone-lattice",
            "--print-args=false",
            model,
            "ark:%s" % lattice_ark,
            "ark:-",
        ],
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )
    # Best path in the character lattice
    p2 = subprocess.Popen(
        [
            "lattice-best-path",
            "--acoustic-scale=%f" % acoustic_scale,
            "ark:-",
            "ark,t:-",
        ],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        stderr=None if verbose else DEV_NULL,
    )
    # Convert integers to symbols (characters)
    p3 = subprocess.Popen(
        ["int2sym.pl", "-f", "2-", syms], stdin=p2.stdout, stdout=subprocess.PIPE
    )
    # Remove multiple repetitions of white spaces
    p4 = subprocess.Popen(
        ["sed", "-r", "s| {wspace}( {wspace})+| {wspace}|g".format(wspace=wspace)],
        stdin=p3.stdout,
        stdout=subprocess.PIPE,
    )
    hyp_txt = tempfile.NamedTemporaryFile()
    # Remove whitespace at the start and end of the sentence
    p5 = subprocess.Popen(
        [
            "awk",
            """
            {
              if ($2 == "%s") { $2 = ""; }
              if ($NF == "%s") { $NF = ""; }
              print;
            }
            """
            % (wspace, wspace),
        ],
        stdin=p4.stdout,
        stdout=hyp_txt,
    )
    p5.communicate()
    cer = compute_err(ref_txt, hyp_txt.name)

    w_ref_txt = get_word_from_char_txt(ref_txt, wspace, char_sep)
    w_hyp_txt = get_word_from_char_txt(hyp_txt, wspace, char_sep)
    wer = compute_err(w_ref_txt.name, w_hyp_txt.name)
    return {"CER": cer, "WER": wer}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-scale", default=1.0, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wspace", default="<space>")
    parser.add_argument("--char-separator", default="")
    parser.add_argument("syms")
    parser.add_argument("model")
    parser.add_argument("lattice_ark")
    parser.add_argument("ref_txt")
    args = parser.parse_args()

    print(
        htr_assessment(
            ref_txt=args.ref_txt,
            lattice_ark=args.lattice_ark,
            model=args.model,
            syms=args.syms,
            wspace=args.wspace,
            acoustic_scale=args.acoustic_scale,
            char_sep=args.char_separator,
            verbose=args.verbose,
        )
    )
