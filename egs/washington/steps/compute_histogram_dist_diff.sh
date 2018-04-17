#!/bin/bash
set -e;
export LC_NUMERIC=C;

BW=0.1;
while [ "${1:0:1}" = "-" ]; do
  case "$1" in
    -w)
      BW="$2"; shift 2;
      ;;
    *)
      echo "Unknown option \"$1\"!" >&2 && exit 1;
  esac;
done;


awk -v bw="$BW" '
function bin(x) {
  return bw * int(x / bw);
}
BEGIN{
  maxb = -1e50;
  minb = +1e50;
  N = 0;
}{
  b = bin($1);
  BC[b]++;
  N++;
  if (b < minb) minb = b;
  if (b > maxb) maxb = b;
}END{
  for (b in BC) {
    print b, BC[b], BC[b] / N;
  }
}' "$@" | sort -gk1;
