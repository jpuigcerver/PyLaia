#!/bin/bash
set -e;

[ $# -ne 1 ] && echo "Usage: ${0##*/} <matrix-rspecifier>" >&2 && exit 1;
copy-matrix --print-args=false "$1" ark,t:- |
gawk '
function logadd(a, b) {
  if (b > a) { t=a; a=b; b=t; }
  z = b - a;
  return a + log(1.0 + exp(z));
}
BEGIN{
  DIM = -1;
  N = 0;
}{
  if ($2 == "[") next;
  if ($NF == "]") d = NF - 1;
  else d = NF;
  if (d == 0) next;
  if (DIM < 0) DIM = d;
  else if (d != DIM) {
    print "Wrong number of dimensions! (expected: "DIM", found: "d")" > "/dev/stderr";
    exit(1);
  }

  for (i = 1; i <= DIM; ++i) {
    if (i in S) {
      S[i] = logadd(S[i], $i);
    } else {
      S[i] = $i;
    }
  }
  N++;
}END{
  for (d = 1; d <= DIM; ++d) {
    printf("%.9e\n", S[d] - log(N));
  }
}' 2> >(egrep -v "exp: argument .+ is out of range")
