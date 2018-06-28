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
