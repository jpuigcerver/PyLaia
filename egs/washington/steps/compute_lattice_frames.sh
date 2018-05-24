#!/bin/bash
set -e;

[ $# -eq 0 ] &&
echo "Usage: ${0##*/} <lattice-ark1> [<lattice-ark2> ...]" >&2 &&
exit 1;

while [ $# -gt 0 ]; do
  lattice-depth-per-frame --print-args=false "ark:$1" ark,t:- |
  awk '{ print $1, NF - 1; }'
  shift 1;
done;
