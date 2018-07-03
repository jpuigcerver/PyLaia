#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export LC_ALL=en_US.utf8;
export LC_NUMERIC=C;

check_all_files data/lang/lines/word/tr_original.txt || exit 1;

mkdir -p data/lang/lines/{char,word};

# 1. Put all words in lowercase.
# 2. Remove ' and ", since these are not relevant for KWS
# 3. Replace start-of-split word delimiter :, with =
# 4. Replace delimiters , . : ;  with whitespace
# 5. Separate [ ( ) ] from the other characters
# 6. Separate <gap/> from other symbols
# 7. Separate <del> and </del> tokens from other symbols and remove them.
# 8. Remove contiguous spaces
[ -s data/lang/lines/word/tr_normalized.txt ] ||
awk '{
  printf("%s", $1);
  for (i = 2; i <= NF; ++i) { printf(" %s", tolower($i));  }
  printf("\n");
}' data/lang/lines/word/tr_original.txt |
sed -r 's|['\''"]||g' |
sed -r 's|^([^ ]+) +:([^ ]+)|\1 =\2|g' |
sed -r 's|[,.:;]| |g' |
sed -r 's/(\[|\()/ \1 /g;s/(\]|\))/ \1 /g' |
sed -r 's|(<gap/>)| \1 |g' |
sed -r 's|(</?del>)| \1 |g;s|</?del>||g;s|£>|£|g' |
sed -r 's| +| |g;s| $||g' |
sort -V > data/lang/lines/word/tr_normalized.txt || {
  echo "ERROR: Creating file \"data/lang/lines/word/tr_normalized.txt\"!" >&2;
  exit 1;
}

# Split words into characters to train the neural network
[ -s data/lang/lines/char/tr_normalized.txt ] ||
awk '{
  printf("%s", $1);
  for (i = 2; i <= NF; ++i) {
    if ($i == "<gap/>" || $i == "<gap>") {
      printf(" <@>");
    } else if ($i == "#") {
      printf(" <#>");
    } else {
      for (j = 1; j <= length($i); ++j) {
        printf(" %s", substr($i, j, 1));
      }
    }
    if (i < NF) {
      printf(" <sp>");
    }
  }
  printf("\n");
}' data/lang/lines/word/tr_normalized.txt \
   > data/lang/lines/char/tr_normalized.txt;

# Prepare table of character symbols for CTC training
[ -s data/lang/syms_ctc.txt ] ||
cut -d\  -f2- data/lang/lines/char/tr_normalized.txt |
tr \  \\n | sort -u | awk 'BEGIN{
  N=0;
  printf("%5-s %d\n", "<ctc>", N++);
  printf("%5-s %d\n", "<sp>", N++);
  printf("%5-s %d\n", "<@>", N++);
  printf("%5-s %d\n", "<#>", N++);
}{
  if ($1 == "<sp>" || $1 == "<@>" || $1 == "<#>") next;
  printf("%5-s %d\n", $1, N++);
}' > data/lang/syms_ctc.txt;

# Split data into test, training and validation partitions.
[ -s data/lang/lines/char/tr.txt -a \
  -s data/lang/lines/char/va.txt ] ||
gawk 'BEGIN{
  srand(1234);
}{
  if (rand() < 0.1) {
    print $0 > "data/lang/lines/char/va.txt";
  } else {
    print $0 > "data/lang/lines/char/tr.txt";
  }
}' data/lang/lines/char/tr_normalized.txt;
