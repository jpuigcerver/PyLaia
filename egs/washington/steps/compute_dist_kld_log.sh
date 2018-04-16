#!/bin/bash
set -e;

[ $# -lt 1 ] && echo "Usage: ${0##*/} <ref> [<hyp>]" >&2 && exit 1;

RF="$1";
shift;

gawk -v RF="$RF" -v PENALTY=1000 '
function abs(x) {
  return x < 0 ? -x : x;
}
function min(a, b) {
  return a < b ? a : b;
}
function max(a, b) {
  return a > b ? a : b;
}
BEGIN{
  N = S = NE = 0;
  while ((getline < RF) > 0) {
    pair = $1" "$2;
    PR[pair] = 1;
    M[pair] = 0;
  }
}{
  pair = $1" "$2;
  if ($3 > 0) {
    if ($3 > 1e-5) {
      print "Wrong log-probability value at line " NR ": " $3 > "/dev/stderr";
    }
    $3 = 0;
  }
  if (pair in PR) {
    M[pair] = 1;
    if ($3 == "-inf") {
      print PENALTY;
      S += PENALTY;
      NE++;
    } else {
      print -$3;
      S += -$3;
    }
    N++;
  }
}END{
  for (pair in PR) {
    if (M[pair] == 0) {
      print PENALTY;
      S += PENALTY;
      N++; NE++;
      print pair > "/dev/stderr";
    }
  }
  if (NE > 0) {
    print "Number of pairs with inf divergence = " NE > "/dev/stderr";
  }
  print "Mean Kullback-Leibler Divergence = " S / N > "/dev/stderr";
}' "$@";
