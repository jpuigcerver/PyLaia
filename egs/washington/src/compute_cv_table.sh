#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

help_message="
Usage: ${0##*/} (phoc|prob_phoc|ctc)
";
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

case "$1" in
  phoc)
    sort=asc;
    ;;
  prob_phoc)
    sort=desc;
    ;;
  ctc)
    sort=desc;
    ;;
  *)
    echo -e "Unknown method: \"$1\"!\n$help_message" >&2 && exit 1;
esac;

check_all_programs SimpleKwsEval || exit 1;

check_all_files -s data/lang/dortmund/cv1_rel_qbe.txt \
                   data/lang/dortmund/cv2_rel_qbe.txt \
                   data/lang/dortmund/cv3_rel_qbe.txt \
                   data/lang/dortmund/cv4_rel_qbe.txt || exit 1;

tmpf="$(mktemp)";
for r in 01 02 03 04 05 06 07 08 09 10; do
  for cv in cv1 cv2 cv3 cv4; do
    ckpt="index/dortmund/$1/$cv/r$r/epoch-160.dat.gz";
    check_all_files -s "$ckpt" || exit 1;
    zcat "$ckpt" |
    SimpleKwsEval \
      --collapse_matches true \
      --trapezoid_integral false \
      --interpolated_precision false \
      --sort "$sort" \
      data/lang/dortmund/${cv}_rel_qbe.txt 2> /dev/null ||
    { cat "$tmpf" >&2 && exit 1; }
  done;
done |
gawk '
function print_head() {
  printf("      | CV1      | CV2      | CV3      | CV4      | Avg\n");
  printf("------+----------+----------+----------+----------+---------\n");
}

function print_row(row) {
  s = 0;
  for (cv = 1; cv <= 4; ++cv) {
    printf(" | %.6f", row[cv]);
    s += row[cv];
  }
  printf(" | %.6f\n", s / 4.0);
}

BEGIN{
 n = 0; cv = 1; r = 1;
}{
  if ($1 == "mAP") { mAP[cv] = $3; sum_mAP[cv] += $3; ++n; }
  else if ($1 == "gAP") { gAP[cv] = $3; sum_gAP[cv] += $3; ++n; }
  else if ($1 == "mNDCG") { mNDCG[cv] = $3; sum_mNDCG[cv] += $3; ++n; }
  else if ($1 == "gNDCG") { gNDCG[cv] = $3; sum_gNDCG[cv] += $3; ++n; }
  if (n % 4 == 0) { ++cv; }

  if (cv == 5) {
    printf("r%02d\n", r);
    printf("===\n");
    print_head();
    printf("%5-s", "mAP"); print_row(mAP);
    printf("%5-s", "gAP"); print_row(gAP);
    printf("%5-s", "mNDCG"); print_row(mNDCG);
    printf("%5-s", "gNDCG"); print_row(gNDCG);
    printf("\n");
    cv = 1;
    ++r;
  }
}END{
  for (cv = 1; cv <= 4; ++cv) {
    sum_mAP[cv] = sum_mAP[cv] / (r - 1);
    sum_gAP[cv] = sum_gAP[cv] / (r - 1);
    sum_mNDCG[cv] = sum_mNDCG[cv] / (r - 1);
    sum_gNDCG[cv] = sum_gNDCG[cv] / (r - 1);
  }
  printf("Avg\n");
  printf("===\n");
  print_head();
  printf("%5-s", "mAP"); print_row(sum_mAP);
  printf("%5-s", "gAP"); print_row(sum_gAP);
  printf("%5-s", "mNDCG"); print_row(sum_mNDCG);
  printf("%5-s", "gNDCG"); print_row(sum_gNDCG);
  printf("\n");
}'

rm -f "$tmpf";
exit 0;
