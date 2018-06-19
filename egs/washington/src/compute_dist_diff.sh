#!/bin/bash
set -e;

[ $# -ne 1 -a $# -ne 2 ] && echo "Usage: ${0##*/} <ref> [<hyp>]" >&2 && exit 1;

RF="$1"; shift;

gawk -v RF="$RF" '
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
  N = S = 0;
  while ((getline < RF) > 0) {
    pair = $1" "$2;
    P[pair] = 1;
    M[pair] = 0;
  }
}{
  pair = $1" "$2;
  prob = max(min($3, 1), 0);
  if (prob != $3) {
    print "Wrong probability value at line " NR ": " $3 > "/dev/stderr";
  }
  M[pair] = 1;
  print P[pair] - prob;
  S += abs(P[pair] - prob);
  N++;
}END{
  for (pair in P) {
    if (M[pair] == 0) {
      print 1.0;
      S += 1.0;
      N++;
    }
  }

  print "Mean Absolute Error = " S / N > "/dev/stderr";
}' "$@";
