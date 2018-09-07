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

for as in $(seq 0.2 0.2 4.0); do
  for ps in $(seq 0.0 0.1 1.5); do
    for cv in cv1 cv2 cv3 cv4; do
      check_all_files -s "$latdir/$cv/ps${ps}/va.lat.ark" \
		         "$fstdir/$cv/chars.txt" \
                         "data/fki/lang/queries/$cv.txt" \
			 "data/fki/lang/kws_refs/$cv/va.txt" || exit 1;
      readarray -t delimiters < \
          <(join -1 1 <(sort -k1b,1 data/fki/lang/delimiters.txt) \
                      <(sort -k1b,1 "$fstdir/$cv/chars.txt") |
            sort -nk2 | awk '{print $2}') || exit 1;
      ./src/fki/compute_kws_metrics_char.py \
	--acoustic-scale="$as" \
	--queries="data/fki/lang/queries/$cv.txt" \
	--index-type=segment \
	"$fstdir/$cv/chars.txt" \
	"data/fki/lang/kws_refs/$cv/va.txt" \
	"$latdir/$cv/ps${ps}/va.lat.ark" \
	"${delimiters[@]}" || exit 1;
    done |
    gawk -v as="${as}" -v ps="${ps}" '
    BEGIN{
      sMAP = 0.0; sGAP = 0.0;
    }{
      if (match($0, "'\''mAP'\'': ([0-9.]+)", mAP) &&
          match($0, "'\''gAP'\'': ([0-9.]+)", gAP)) {
        sMAP += mAP[1];
        sGAP += gAP[1];
      } else {
        print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
      }
    }END{
      printf("%.1f %.1f %.1f %.1f\n",
             as, ps, sMAP * 100.0 / NR, sGAP * 100.0 / NR);
    }'
  done;
  echo "";
done;
