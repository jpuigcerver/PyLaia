#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";

acoustic_scale=1.0;
wspace="<space>";
char_sep="";
help_message="
Usage: ${0##*/} <fst_dir> <lats_dir>
Options:
  --acoustic_scale : (type = float, default = $acoustic_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

check_all_files -s "$1/model" "$1/chars.txt" \
                   "$2/va.lat.ark" "$2/te.lat.ark" \
                   data/kws_line/lang/queries/iam_old_papers_queries.txt || exit 1;

idxs=("$(mktemp)" "$(mktemp)");
for p in va te; do
  tmp="$(mktemp)";
  date;
  ./src/kws_line/lattice_decode.sh \
    --acoustic_scale "$acoustic_scale" \
    --wspace "$wspace" \
    --char_sep "$char_sep" \
    "$1" "$2/$p.lat.ark" |
  awk -v reffile="data/kws_line/lang/kws_refs/$p.txt" '
  BEGIN{
    while((getline < reffile) > 0) {
      REF[$2" "$1] = 1;
    }
  }{
     for (i = 2; i <= NF; ++i) {
       pair = $1" "$i;
       if (pair in REF) {
         print $1, $i, 1, 1.0;
         HYP[pair] = 1;
       } else {
         print $1, $i, 0, 1.0;
       }
       IDX++;
     }
  }END{
    print "INDEXED SIZE = "IDX > "/dev/stderr";
    for (pair in REF) {
      if (!(pair in HYP)) {
        print pair, 1, "-inf";
      }
    }
  }' | sort -u > "$tmp";
  date;
  kws-assessment-joan \
    -a -m -t -w data/kws_line/lang/queries/iam_old_papers_queries.txt "$tmp" |
  awk '{ if(match($0, /AP = ([0-9.]+)/, A)) print $1, A[1]; }'
  rm $tmp;
done;
