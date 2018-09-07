#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

num_epochs=120;
use_distortions=false;
help_message="
Usage: ${0##*/} <output_dir>
  --num_epochs       : (type = integer, default = $num_epochs)
  --use_distortions  : (type = boolean, default = $use_distortions)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

check_all_files -s data/kws_line/lang/queries/iam_old_papers_queries.txt \
                   data/kws_line/lang/delimiters.txt \
                   data/kws_line/lm/char_8gram/chars.txt \
                   data/kws_line/lang/kws_refs/va.txt \
                   data/kws_line/lang/kws_refs/te.txt || exit 1;

mkdir -p "$1" || exit 1;

if [ "$use_distortions" = true ]; then
  suffix="_distortions";
else
  suffix="";
fi;

readarray -t delimiters < \
    <(join -1 1 <(sort -k1b,1 data/kws_line/lang/delimiters.txt) \
                <(sort -k1b,1 data/kws_line/lm/char_8gram/chars.txt) |
      sort -nk2 | awk '{print $2}') || exit 1;

for tr in 2054 4108 6161 8216 11504; do
  exp="resize_h128_tr_${tr}_epochs${num_epochs}${suffix}";
  latdir="data/kws_line/lats/$exp/char_8gram";
  check_all_dirs "$latdir" || exit 1;
  check_all_files -s "$latdir/ps0.0/te.lat.ark" \
                     "$latdir/ps0.0/va.lat.ark" \
                     "$latdir/ps1.0/te.lat.ark" \
                     "$latdir/ps1.0/va.lat.ark" || exit 1;

  logfile="$1/tune_acoustic_prior_scale_segment_$exp.log";
  [ -s "$logfile" ] ||
  ( ./src/kws_line/tune_metric_kws_char.py \
    --queries=data/kws_line/lang/queries/iam_old_papers_queries.txt \
    --index-type=segment \
    data/kws_line/lm/char_8gram/chars.txt \
    data/kws_line/lang/kws_refs/va.txt \
    "$latdir/ps{prior_scale:.1f}/va.lat.ark" \
    "${delimiters[@]}" 2>&1 |
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
  ./src/kws_line/compute_kws_metrics_char.py \
    --acoustic-scale="${params[0]}" \
    --queries=data/kws_line/lang/queries/iam_old_papers_queries.txt \
    --index-type=segment \
    data/kws_line/lm/char_8gram/chars.txt \
    data/kws_line/lang/kws_refs/te.txt \
    "$latdir/ps${params[1]}/te.lat.ark" \
    "${delimiters[@]}" |
  gawk -v tr="$tr" -v ne="$num_epochs" \
       -v as="${params[0]}" -v ps="${params[1]}" \
       -v va_map="${va_map_gap[0]}" -v va_gap="${va_map_gap[1]}" \
  '{
    if (match($0, "'\''mAP'\'': ([0-9.]+)", mAP) &&
        match($0, "'\''gAP'\'': ([0-9.]+)", gAP)) {
      printf("%5-d %3-d %.9f %.9f %.9f %.9f (as = %.2f  ps = %.1f)\n",
             tr, ne, va_map, va_gap, mAP[1], gAP[1], as, ps);
    } else {
      print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
    }
  }'
done;
