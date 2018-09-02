#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";
export LC_LANGUAGE=en_US.utf8;

overwrite=false;
height=120;
help_message="
Usage: ${0##*/} [options]
Options:
  --height     : (type = integer, default = $height)
                 Scale lines to have this height, keeping the aspect ratio
                 of the original image.
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

imgdir=data/fki/washingtondb-v1.0/data/line_images_normalized;
txt=data/fki/washingtondb-v1.0/ground_truth/transcription.txt;

check_all_files -s "$txt" || exit 1;
check_all_dirs "$imgdir" || exit 1;

# Obtain character-level transcripts.
mkdir -p data/fki/lang/char;
[[ "$overwrite" = false && -s data/fki/lang/char/transcript.txt ]] ||
gawk '{
  x=$1; y=$2;
  gsub(/-/, " ", y);
  gsub(/\|/, " <space> ", y);
  print x, y;
}' "$txt" |
gawk '{
  printf("%s", $1);
  for (i=2;i<=NF;++i) {
    if ($i == "s_bl")      printf(" (");
    else if ($i == "s_br") printf(" )");
    else if ($i == "s_cm") printf(" ,");
    else if ($i == "s_et") printf(" &");
    else if ($i == "s_lb") printf(" £");
    else if ($i == "s_mi") printf(" -");
    else if ($i == "s_pt") printf(" .");
    else if ($i == "s_qo") printf(" :");
    else if ($i == "s_qt") printf(" '\''");
    else if ($i == "s_s")  printf(" s");
    else if ($i == "s_sl") printf(" /");
    else if ($i == "s_sq") printf(" ;");
    else if ($i == "s_GW") printf(" G . W .");
    else if (match($i, /^s_(.+)$/, A)) {
      for (j = 1; j <= length(A[1]); ++j) {
        printf(" %s", substr(A[1], j, 1));
      }
    }
    else printf(" %s", $i);
  }
  printf("\n");
}' |
sed -r 's| ([\(\)]) | <space> \1 <space> |g' |
sed -r 's|<space> (<space> )*|<space> |g' \
    > data/fki/lang/char/transcript.txt || exit 1;

# Obtain word level transcripts
mkdir -p data/fki/lang/word;
[[ "$overwrite" = false && -s data/fki/lang/word/transcript.txt ]] ||
gawk '{
  printf("%s ", $1);
  for (i = 2; i <= NF; ++i) {
    if ($i == "<space>") printf(" ");
    else printf("%s", $i);
  }
  printf("\n");
}' data/fki/lang/char/transcript.txt \
    > data/fki/lang/word/transcript.txt || exit 1;

# Get list of characters for CTC training
[[ "$overwrite" = false && -s data/fki/lang/char/syms_ctc.txt ]] ||
cut -d\  -f2- data/fki/lang/char/transcript.txt |
tr \  \\n |
sort -uV |
gawk 'BEGIN{
  N = 0;
  printf("%-12s %d\n", "<ctc>", N++);
  printf("%-12s %d\n", "<space>", N++);
}$1 !~ /<space>/{
  printf("%-12s %d\n", $1, N++);
}' > data/fki/lang/char/syms_ctc.txt || exit 1;

# Create reference transcript files for each CV set.
for cv in cv1 cv2 cv3 cv4; do
  mkdir -p "data/fki/lang/char/$cv" "data/fki/lang/word/$cv";
  for p in test train valid; do
    for c in char word; do
      [[ "$overwrite" = false && -s "data/fki/lang/$c/$cv/${p:0:2}.txt" ]] ||
      join -1 1 \
	   <(sort "data/fki/lang/$c/transcript.txt") \
	   <(sort "data/fki/washingtondb-v1.0/sets/$cv/$p.txt") |
      sort -V > "data/fki/lang/$c/$cv/${p:0:2}.txt" || exit 1;
    done;
  done;
done;

mkdir -p data/fki/lang/queries;
for cv in cv1 cv2 cv3 cv4; do
  [[ "$overwrite" = false && -s "data/fki/lang/queries/$cv.txt" ]] ||
  gawk -F- '{
    for (i=1;i<=NF;++i) {
      if ($i == "s_bl")      printf("(");
      else if ($i == "s_br") printf(")");
      else if ($i == "s_cm") printf(",");
      else if ($i == "s_et") printf("&");
      else if ($i == "s_lb") printf("£");
      else if ($i == "s_mi") printf("-");
      else if ($i == "s_pt") printf(".");
      else if ($i == "s_qo") printf(":");
      else if ($i == "s_qt") printf("'\''");
      else if ($i == "s_s")  printf("s");
      else if ($i == "s_sl") printf("/");
      else if ($i == "s_sq") printf(";");
      else if ($i == "s_GW") printf("G.W.");
      else if (match($i, /^s_(.+)$/, A)) {
        for (j = 1; j <= length(A[1]); ++j) {
          printf("%s", substr(A[1], j, 1));
        }
      }
      else printf("%s", $i);
    }
    printf("\n");
  }' data/fki/washingtondb-v1.0/sets/$cv/keywords.txt \
      > "data/fki/lang/queries/$cv.txt";
done;

exit 0;
