#!/bin/bash
set -e;

[ $# -lt 1 ] && echo "Usage: ${0##*/} [-nw] <map> [<file1> ...]" >&2 && exit 1;
if [ $1 = "-nw" ]; then nw=1; shift 1; else nw=0; fi;
syms="$1"; shift 1;

awk -v nw="$nw" 'BEGIN{
  N = 0;
}{
  w = $1;
  for (i = 2; i <= NF; ++i) {
    w = w" "$i;
  }
  CNT[w]++;
  N++;
}END{
  NS = 0;
  for (w in CNT) {
    n = split(w, W, " ");
    if (nw) {
      print 0, ++NS, W[1];
    } else {
      print 0, ++NS, W[1], log(N) - log(CNT[w]);
    }
    for (i = 2; i <= n; ++i) {
      print NS, ++NS, W[i];
    }
    print NS;
  }
}' $@ |
fstcompile --acceptor --isymbols="$syms" --osymbols="$syms" |
if [ "$nw" -eq 1 ]; then
  fstdeterminizestar --print-args=false;
else
  fstdeterminizestar --print-args=false --use-log;
fi | fstminimize;
