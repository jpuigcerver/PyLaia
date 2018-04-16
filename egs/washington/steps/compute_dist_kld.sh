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
  prob = max(min($3, 1), 0);
  if (prob != $3) {
    print "Wrong probability value at line " NR ": " $3 > "/dev/stderr";
  }
  if (pair in PR) {
    M[pair] = 1;
    if (prob > 0) {
      print -log(prob);
      S += -log(prob);
      N++;
    } else {
      NE++;
    }
  }
}END{
  for (pair in PR) {
    if (M[pair] == 0) {
      NE++;
    }
  }
  if (NE > 0) {
    print "Number of pairs with inf divergence = " NE > "/dev/stderr";
  }
  print "Mean Kullback-Leibler Divergence = " S / N > "/dev/stderr";
}' "$@";
