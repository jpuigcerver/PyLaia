#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

[ -s data/gw_20p_wannot/3090309_boxes.txt ] ||
{ echo "Missing \"data/gw_20p_wannot/*_boxes.txt\" files!" >&2 && exit 1; }

which convert identify &> /dev/null ||
{ echo "ImageMagick does not seem installed in your system!" >&2 && exit 1; }

mkdir -p data/imgs/dortmund;
[ -s data/imgs/dortmund/3090309_67x46+1723+2991.png ] ||
  gawk '$0 ~ /^[ ]*<spot /{
    if (!match($0, /h="([0-9]+)"/, H) ||
        !match($0, /w="([0-9]+)"/, W) ||
        !match($0, /y="([0-9]+)"/, Y) ||
        !match($0, /x="([0-9]+)"/, X) ||
        !match($0, /image="([0-9]+).png"/, PAGE) ||
        !match($0, /word="([a-z0-9]+)"/, WORD)) {
      print "Line "NR" in file "FILENAME" was not expected!:" > "/dev/stderr";
      print $0 > "/dev/stderr";
    }
    IMG_ID = sprintf("%s_%dx%d+%d+%d", PAGE[1], W[1], H[1], X[1], Y[1]);
    CMD = sprintf("convert data/gw_20p_wannot/%s.tif -crop %dx%d+%d+%d",
                  PAGE[1], W[1], H[1], X[1], Y[1]);
    CMD = sprintf("%s +repage data/imgs/dortmund/%s.png", CMD, IMG_ID);
    system(CMD);
  }' data/dortmund_gt/gw_cv1_test.xml data/dortmund_gt/gw_cv1_train.xml;

# Extract ground-truth information
mkdir -p data/lang/dortmund/word;
for f in data/dortmund_gt/*.xml; do
  A=($(echo "$(basename $f)" | sed -r 's/^gw_(cv[0-9])_(te|tr).+$/\1 \2/g'));
  [ -s "data/lang/dortmund/${A[1]}_${A[2]}.txt" ] ||
  gawk '$0 ~ /^[ ]*<spot /{
    if (!match($0, /h="([0-9]+)"/, H) ||
        !match($0, /w="([0-9]+)"/, W) ||
        !match($0, /y="([0-9]+)"/, Y) ||
        !match($0, /x="([0-9]+)"/, X) ||
        !match($0, /image="([0-9]+).png"/, PAGE) ||
        !match($0, /word="([a-z0-9]+)"/, WORD)) {
      print "Line "NR" in file "FILENAME" was not expected!:" > "/dev/stderr";
      print $0 > "/dev/stderr";
    }
    IMG_ID = sprintf("%s_%dx%d+%d+%d", PAGE[1], W[1], H[1], X[1], Y[1]);
    printf("%-25s %s\n", IMG_ID, WORD[1]);
  }' "$f" > "data/lang/dortmund/word/${A[0]}_${A[1]}.txt";
done;

mkdir -p data/lang/dortmund/char;
for f in data/lang/dortmund/word/cv*.txt; do
  [ -s "${f/word/char}" ] ||
  awk '{
    printf("%-25s", $1);
    for (j=1; j <= length($2); ++j) {
      printf(" %s", substr($2, j, 1));
    }
    printf("\n");
  }' "$f" > "${f/word/char}";
done;

for cv in cv1 cv2 cv3 cv4; do
  [ -s data/lang/dortmund/${cv}_rel_qbe.txt ] ||
  awk '{
    if ($2 in IMGS) {
      IMGS[$2] = IMGS[$2]" "$1;
    } else {
      IMGS[$2] = $1;
    }
  }END{
    for (w in IMGS) {
      n = split(IMGS[w], A, " ");
      for (i=1; i<=n; ++i) {
        for (j=i+1; j<=n; ++j) {
          print A[i], A[j];
          print A[j], A[i];
        }
      }
    }
  }' data/lang/dortmund/word/${cv}_te.txt \
      > data/lang/dortmund/${cv}_rel_qbe.txt;
done;

# Prepare symbols table for CTC training
[ -s data/lang/dortmund/syms_ctc.txt ] ||
awk '{ for(i=2; i <= NF; ++i) print $i; }' \
  data/lang/dortmund/char/cv1_{te,tr}.txt |
  sort -V | uniq |
  awk 'BEGIN{
    N=0;
    printf("%-8s %d\n", "<ctc>", N++);
  }{
    printf("%-8s %d\n", $1, N++);
  }' > data/lang/dortmund/syms_ctc.txt;

# Prepare symbols table for PHOC training
[ -s data/lang/dortmund/syms_phoc.txt ] ||
awk 'NR>1{ printf("%-5s %d\n", $1, NR - 2); }' \
    data/lang/dortmund/syms_ctc.txt > data/lang/dortmund/syms_phoc.txt;
