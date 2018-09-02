#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

help_message="
Usage: ${0##*/} <output_dir>
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

check_all_files -s data/kws_line/lang/queries/iam_old_papers_queries.txt \
                   data/kws_line/lang/kws_refs/va.txt \
                   data/kws_line/lang/kws_refs/te.txt || exit 1;

mkdir -p "$1" || exit 1;

for n in 1 2 3 4 5; do
  latdir="data/kws_line/lats/resize_h128/word_${n}gram_50k";
  check_all_dirs "$latdir" || exit 1;
  check_all_files -s "data/kws_line/lm/word_${n}gram_50k/words.txt" || exit 1;

  logfile="$1/tune_acoustic_prior_scale_segment_resize_h128_word_${n}gram_50k.log";
  [ -s "$logfile" ] ||
  ( ./src/kws_line/tune_metric_kws_word.py \
    --queries=data/kws_line/lang/queries/iam_old_papers_queries.txt \
    --index-type=segment \
    data/kws_line/lm/word_${n}gram_50k/words.txt \
    data/kws_line/lang/kws_refs/va.txt \
    "$latdir/ps{prior_scale:.1f}/va.lat.ark" 2>&1 |
    tee "$logfile"; )  || exit 1;

  readarray -t params < \
      <(tail -n1 "$logfile" |
        gawk '{
          if (match($0, "'\''acoustic_scale'\'': ([0-9.]+)", AM) &&
              match($0, "'\''prior_scale'\'': ([0-9.]+)", PM)) {
            printf("%.2f\n", AM[1]);
            printf("%.1f\n", PM[1]);
          } else {
            print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
          }
        }') || exit 1;
  readarray -t va_map_gap < \
      <(grep "acoustic_scale = ${params[0]}  prior_scale = ${params[1]}" \
	     "$logfile" | gawk '{ print $9; print $12; }') || exit 1;

  check_all_files -s "$latdir/ps${params[1]}/te.lat.ark" || exit 1;
  ./src/kws_line/compute_kws_metrics_word.py \
    --acoustic-scale="${params[0]}" \
    --queries=data/kws_line/lang/queries/iam_old_papers_queries.txt \
    --index-type=segment \
    data/kws_line/lm/word_${n}gram_50k/words.txt \
    data/kws_line/lang/kws_refs/te.txt \
    "$latdir/ps${params[1]}/te.lat.ark" |
  gawk -v n="$n" -v as="${params[0]}" -v ps="${params[1]}" \
       -v va_map="${va_map_gap[0]}" -v va_gap="${va_map_gap[1]}" \
  '{
    if (match($0, "'\''mAP'\'': ([0-9.]+)", mAP) &&
        match($0, "'\''gAP'\'': ([0-9.]+)", gAP)) {
      printf("%2-d %.9f %.9f %.9f %.9f (as = %.2f  ps = %.1f)\n",
             n, va_map, va_gap, mAP[1], gAP[1], as, ps);
    } else {
      print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
    }
  }'
done;
