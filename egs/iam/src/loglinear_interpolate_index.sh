#!/bin/bash
set -e;
export LC_NUMERIC=C;

log_cost=0;
log_index=0;
weight=0.5;
while [ "${1:0:1}" = "-" ]; do
  case "$1" in
    -w)
      weight="$2";
      shift 2;
      ;;
    -l)
      log_index=1;
      shift 1;
      ;;
    -c)
      log_cost=1;
      shift 1;
      ;;
    *)
      echo "Unknown option: $1" >&2 && exit 1;
  esac;
done;
[ $# -lt 1 -o $# -gt 2 ] &&
echo "Usage: ${0##*/} [-c | -l] [-w weight] <index1> [<index2>]" >&2 &&
exit 1;

index1="$1";
shift 1;
gawk -v f="$index1" -v w="$weight" -v l="$log_index" -v c="$log_cost" '
function logmul(a, b) {
  if(a == "-inf") return b;
  if(b == "-inf") return a;
  return a + b;
}
BEGIN{
  while ((getline < f) > 0) {
    pair = $1" "$2;
    if (pair in SA) {
      print "WARNING: Pair \""pair"\" was repeated in index1" > "/dev/stderr";
    } else {
      if ($3 == "inf") { SA[pair] = "-inf"; }
      else if ($3 == "-inf") { SA[pair] = "-inf"; }
      else if (c) { SA[pair] = w * (-$3); }
      else if (l) { SA[pair] = w * $3; }
      else { SA[pair] = w * log($3); }
    }
  }
}{
  pair = $1" "$2;
  if ($3 == "inf" || $3 == "-inf") { sB = "-inf"; }
  else if (c) { sB = (1.0 - w) * (-$3); }
  else if (l) { sB = (1.0 - w) * $3; }
  else { sB = (1.0 - w) * log($3); }

  if (pair in SA) {
    if (pair in DONE) {
      print "WARNING: Pair \""pair"\" was repeated in index2" > "/dev/stderr";
    } else {
      print pair, logmul(SA[pair], sB);
      DONE[pair] = 1;
    }
  } else {
    print "WARNING: Pair \""pair"\" not found in index1" > "/dev/stderr";
    print pair, sB;
  }
}END{
  for (pair in SA) {
    if (!(pair in DONE)) {
      print "WARNING: Pair \""pair"\" not found in index2" > "/dev/stderr";
      print pair, SA[pair];
    }
  }
}' "$@";
