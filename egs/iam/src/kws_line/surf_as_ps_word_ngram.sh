#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

help_message="
Usage: ${0##*/} <lats_dir> <fst_dir>
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

latdir="$1";
fstdir="$2";
check_all_dirs "$latdir" "$fstdir" || exit 1;
check_all_files -s data/kws_line/lang/queries/iam_old_papers_queries.txt \
                   data/kws_line/lang/kws_refs/va.txt \
		   "$fstdir/words.txt" || exit 1;

for as in $(seq 0.2 0.2 4.0); do
  for ps in $(seq 0.0 0.1 1.0); do
    check_all_files -s "$latdir/ps${ps}/va.lat.ark" || exit 1;
    ./src/kws_line/compute_kws_metrics_word.py \
      --acoustic-scale="$as" \
      --queries=data/kws_line/lang/queries/iam_old_papers_queries.txt \
      --index-type=segment \
      "$fstdir/words.txt" \
      data/kws_line/lang/kws_refs/va.txt \
      "$latdir/ps${ps}/va.lat.ark" |
    gawk -v as="${as}" -v ps="${ps}" '
    {
      if (match($0, "'\''mAP'\'': ([0-9.]+)", mAP) &&
          match($0, "'\''gAP'\'': ([0-9.]+)", gAP)) {
        printf("%.1f %.1f %.1f %.1f\n",
               as, ps, mAP[1] * 100.0, gAP[1] * 100.0);
      } else {
        print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
      }
    }'
  done;
  echo "";
done;
