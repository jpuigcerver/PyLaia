#!/usr/bin/env bash
set -e;

scale=1;
if [ "$1" == "--scale" ]; then
  scale="$2";
  shift 2 || { echo "ERROR: Missing --scale parameter!" >&2 && exit 1; }
fi;
[ $# -ne 3 ] &&
echo "Usage: ${0##*/} [--scale s] <priors> <mat-rspecifier> <mat-wspecifier>" >&2 &&
exit 1;

copy-matrix --print-args=false "$2" ark,t:- |
gawk -v priors="$1" -v scale="$scale" 'BEGIN{
  DIM = 0;
  while( (getline < priors) > 0 ) { P[++DIM] = $1; }
}{
  if ($2 == "[") {
    print;
    next;
  }

  if ($NF == "]") d = NF - 1;
  else d = NF;
  if (d != DIM) {
    print "Wrong number of dimensions! (expected: "DIM", found: "d")" > "/dev/stderr";
    exit(1);
  }


  for (i = 1; i <= DIM; ++i) {
    $i = $i - scale * P[i];
    printf(" %.6e", $i);
  }

  if ($NF == "]") {
    printf(" ]\n");
  } else {
    printf("\n");
  }
}' | copy-matrix --print-args=false ark,t:- "$3";
