#!/bin/bash
set -e;

[ $# -ne 3 ] &&
echo "Usage: ${0##*/} <reference> <hypothesis> <output>" >&2 && exit 1;

awk -v relf="$1" -v outf="$3" -v w=0.1 'BEGIN{
  while ((getline < relf) > 0) {
    REL[$1" "$2] = 1;
  }
  NB = int(sprintf("%.0g", 1 + 1 / w));
}{
  bin = int(sprintf("%.0g", $3 / w));
  if ($1" "$2 in REL) {
    HI1[bin] = HI1[bin] + 1;
  } else {
    HI0[bin] = HI0[bin] + 1;
  }
}END{
  T0=0; for (bin in HI0) { T0 += HI0[bin]; }
  T1=0; for (bin in HI1) { T1 += HI1[bin]; }
  HI0[NB - 1] += HI0[NB];
  HI1[NB - 1] += HI1[NB];

  for (b = 0; b < NB; ++b) {
    h0 = (b in HI0 ? HI0[b] : 0);
    h1 = (b in HI1 ? HI1[b] : 0);
    print b * w, h0 / T0, h0 > outf".0";
    print b * w, h1 / T1, h1 > outf".1";
  }
}' "$2";
