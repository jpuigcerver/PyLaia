#!/usr/bin/env bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;


check_all_dirs data/gw_20p_wannot || exit 1;
check_all_files -s data/gw_20p_wannot/annotations.txt \
		   data/gw_20p_wannot/file_order.txt || exit 1;

mkdir -p data/almazan_lines || exit 1;

mkdir -p data/almazan_lines/lang/word;

# Extract text lines from the gw_20p_wannot data
[ -s data/almazan_lines/lang/word/original.txt ] ||
./src/extract_gw_20p_wannot_lines.py \
  data/gw_20p_wannot data/almazan_lines/images/original \
  > data/almazan_lines/lang/word/original.txt || exit 1;

# Normalize text as done by Almazan
# (lowercase, remove characters not in [0-9a-z])
[ -s data/almazan_lines/lang/word/normalized.txt ] ||
cat data/almazan_lines/lang/word/original.txt |
gawk '{
 line = $1;
 $1 = "";
 txt = gensub(/[.,:;'\''&£/-]/, "", "g", $0);
 print line, tolower(txt);
}' | sed -r 's|  +| |g;s| $||g' \
    > data/almazan_lines/lang/word/normalized.txt || exit 1;

# Prepare character-level reference for all lines.
mkdir -p data/almazan_lines/lang/char;
for p in normalized original; do
  [ -s data/almazan_lines/lang/char/$p.txt ] ||
  awk '{
    printf("%s", $1);
    for (i=2;i<=NF;++i) {
      for(j=1;j<=length($i);++j) {
        printf(" %s", substr($i, j, 1));
      }
      if (i < NF) { printf(" <sp>"); }
    }
    printf("\n");
  }' data/almazan_lines/lang/word/$p.txt \
      > data/almazan_lines/lang/char/$p.txt || exit 1;
done;

# Prepare list of CTC symbols.
[ -s data/almazan_lines/lang/syms_ctc_normalized.txt ] ||
cut -d\  -f2- data/almazan_lines/lang/char/normalized.txt | tr \  \\n |
sort -u |
gawk 'BEGIN{
  N = 0;
  print "<ctc>", N++;
  print "<sp>", N++;
}$1 != "<sp>"{
  print $1, N++;
}' > data/almazan_lines/lang/syms_ctc_normalized.txt || exit 1;
