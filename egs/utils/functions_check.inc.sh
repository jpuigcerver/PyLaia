#!/usr/bin/env bash

function check_all_dirs () {
  for d in "$@"; do
    [ ! -d "$d" ] && \
    echo "ERROR: Directory \"$d\" does not exist!" >&2 && return 1;
  done;
  return 0;
}

function check_all_files () {
  local s=0;
  if [ "$1" = "-s" ]; then s=1; shift; elif [ "$1" == "--" ]; then shift; fi;
  for f in "$@"; do
    [[ ! ( ( "$s" -eq 0 && -f "$f" ) || ( "$s" -eq 1 && -s "$f" ) ) ]] && \
    echo "ERROR: File \"$f\" does not exist!" >&2 && return 1;
  done;
  return 0;
}

function check_all_programs () {
  for p in "$@"; do
    ! which "$p" &> /dev/null && \
    echo "ERROR: Program \"$p\" was not found in your PATH." && \
    return 1;
  done;
  return 0;
}

function check_textFeats () {
  ! which textFeats &> /dev/null && \
  echo "ERROR: Program \"textFeats\" was not found in your PATH." \
       "Please, download it from https://github.com/mauvilsa/textfeats" >&2 && \
  return 1;
  return 0;
}

function check_imagemagick () {
  ! which convert &> /dev/null && \
  echo "ERROR: Program \"convert\" was not found in your PATH." \
       "Please, install ImageMagick in your computer" >&2 && \
  return 1;
  return 0;
}

function check_imgtxtenh () {
  ! which imgtxtenh &> /dev/null && \
  echo "ERROR: Program \"imgtxtenh\" was not found in your PATH." \
       "Please, download it from https://github.com/mauvilsa/imgtxtenh" >&2 && \
  return 1;
  return 0;
}

function check_opengrm () {
  for p in farcompilestrings ngramcount ngrammake ngramperplexity; do
    ! which "$p" &> /dev/null && \
    echo "ERROR: Program \"$p\" was not found in your PATH." \
         "Please, download OpenGrm from http://www.opengrm.org/" >&2 && \
    return 1;
  done;
  return 0;
}

function check_kaldi () {
  for p in add-self-loops fstaddselfloops fstarcsort fstcompile fstcompose \
           fstcomposecontext fstdeterminizestar fstminimizeencoded fstpushspecial \
           fstrelabel fstrmepslocal fstrmsymbols fsttablecompose \
           latgen-faster-mapped-parallel make-h-transducer \
           compute-wer compute-wer-bootci; do
    ! which "$p" &> /dev/null && \
    echo "ERROR: Program \"$p\" was not found in your PATH." \
         "Please, download Kaldi from http://kaldi-asr.org/" >&2 && \
    return 1;
  done;
  return 0;
}

function confirm_overwrite_all_files () {
  for f in "$@"; do
    if [ -f "$f" ]; then
        read -p "File \"$f\" already exists. Do you want to overwrite it? [y/n] " -n 1 -r;
        echo "";
        [[ ! ( "$REPLY" =~ ^[Yy]$ ) ]] && { echo "$REPLY"; return 1; }
    fi;
  done;
  return 0;
}

function check_running_from_parent () {
  local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";
  [ "$PWD" == "$dir" ] || {
    echo "Please, run this script from the experiment top directory!" >&2 && \
    return 1;
  }
  return 0;
}
