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

check_all_files -s data/kws_line/lang/char/va.txt \
                   data/kws_line/lang/char/te.txt || exit 1;

mkdir -p "$1" || exit 1;

for n in 3 4 5 6 7 8 9; do
  latdir="data/kws_line/lats/resize_h128/char_${n}gram";
  check_all_dirs "$latdir" || exit 1;
  check_all_files -s "data/kws_line/lm/char_${n}gram/chars.txt" \
                     "data/kws_line/lm/char_${n}gram/model" || exit 1;

  logfile="$1/tune_acoustic_prior_scale_htr_char_${n}gram.log";
  [ -s "$logfile" ] ||
  ( ./src/kws_line/tune_metric_htr.py \
    "data/kws_line/lm/char_${n}gram/chars.txt" \
    "data/kws_line/lm/char_${n}gram/model" \
    "$latdir/ps{prior_scale:.1f}/va.lat.ark" \
    data/kws_line/lang/char/va.txt 2>&1 |
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
  readarray -t va_cer_wer < \
      <(grep "acoustic_scale = ${params[0]}  prior_scale = ${params[1]}" \
	     "$logfile" | gawk '{ print $9; print $12; }') || exit 1;

  check_all_files -s "$latdir/ps${params[1]}/te.lat.ark" || exit 1;
  ./src/kws_line/compute_metric_htr.py \
    --acoustic-scale="${params[0]}" \
    "data/kws_line/lm/char_${n}gram/chars.txt" \
    "data/kws_line/lm/char_${n}gram/model" \
    "$latdir/ps${params[1]}/te.lat.ark" \
    data/kws_line/lang/char/te.txt |
  gawk -v n="$n" -v as="${params[0]}" -v ps="${params[1]}" \
       -v va_cer="${va_cer_wer[0]}" -v va_wer="${va_cer_wer[1]}" \
  '{
    if (match($0, "'\''CER'\'': ([0-9.]+)", CER) &&
        match($0, "'\''WER'\'': ([0-9.]+)", WER)) {
      printf("%2-d %.2f %.2f %.2f %.2f (as = %.2f  ps = %.1f)\n",
             n, va_cer, va_wer, CER[1], WER[1], as, ps);
    } else {
      print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
    }
  }'
done;
