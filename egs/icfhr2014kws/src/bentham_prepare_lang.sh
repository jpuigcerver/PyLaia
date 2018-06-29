#!/usr/bin/env bash
set -e;
export LC_ALL=en_US.utf8;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

check_all_dirs data/duth/TRACK_II_Bentham_Dataset data/bentham/imgs/lines || exit 1;
check_all_programs xmlstarlet || exit 1;

mkdir -p data/bentham/lang/{char,word};

# Extract ground-truth from the PAGE XML files
[ -s data/bentham/lang/word/original.txt ] ||
find data/bentham/imgs/lines -name "*.xml" |
xargs xmlstarlet sel -t -m '//_:TextLine' -v '../../@imageFilename' -o ' ' \
                -v @id -o " " -v _:TextEquiv/_:Unicode -n |
sed -r 's|^([0-9_]+)\.jpg |\1.|g' |
perl -MHTML::Entities -pe 'decode_entities($_);' |
sed -r 's| +| |g;s| $||g' |
awk 'NF > 0' |
sort -V > data/bentham/lang/word/original.txt;


# 1. Put all words in lowercase.
# 2. Remove ' and ", since these are not relevant for KWS
# 3. Replace start-of-split word delimiter :, with =
# 4. Replace delimiters , . : ;  with whitespace
# 5. Separate [ ( ) ] from the other characters
# 6. Separate <gap/> from other symbols
# 7. Separate <del> and </del> tokens from other symbols and remove them.
# 7. Remove contiguous spaces
# 8. Put page name and line ID together with a .  (recover from step 4)
[ -s data/bentham/lang/word/normalized.txt ] ||
awk '{
  printf("%s", $1);
  for (i = 2; i <= NF; ++i) { printf(" %s", tolower($i));  }
  printf("\n");
}' data/bentham/lang/word/original.txt |
sed -r 's|['\''"]||g' |
sed -r 's|^([^ ]+) +:([^ ]+)|\1 =\2|g' |
sed -r 's|[,.:;]| |g' |
sed -r 's/(\[|\()/ \1 /g;s/(\]|\))/ \1 /g' |
sed -r 's|(<gap/>)| \1 |g' |
sed -r 's|(</?del>)| \1 |g;s|</?del>||g;s|£>|£|g' |
sed -r 's| +| |g;s| $||g' |
sed -r 's|^([^ ]+) ([^ ]+)|\1.\2|g' > data/bentham/lang/word/normalized.txt;


# Split words into characters to train the neural network
[ -s data/bentham/lang/char/normalized.txt ] ||
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
}' data/bentham/lang/word/normalized.txt \
   > data/bentham/lang/char/normalized.txt;


# Prepare table of character symbols for CTC training
[ -s data/bentham/lang/syms_ctc.txt ] ||
cut -d\  -f2- data/bentham/lang/char/normalized.txt |
tr \  \\n | sort -u | awk 'BEGIN{
  N=0;
  printf("%5-s %d\n", "<ctc>", N++);
  printf("%5-s %d\n", "<sp>", N++);
  printf("%5-s %d\n", "<@>", N++);
  printf("%5-s %d\n", "<#>", N++);
}{
  if ($1 == "<sp>" || $1 == "<@>" || $1 == "<#>") next;
  printf("%5-s %d\n", $1, N++);
}' > data/bentham/lang/syms_ctc.txt;


# Split data into test, training and validation partitions.
[ -s data/bentham/lang/char/te.txt -a \
  -s data/bentham/lang/char/tr.txt -a \
  -s data/bentham/lang/char/va.txt ] || {
  tmp="$(mktemp)";
  find data/duth/TRACK_II_Bentham_Dataset -name "*.tif" |
  awk '{ match($0, /^.*\/([0-9_]+)\.tif$/, a); print a[1]; }' |
  sort > "$tmp";

  gawk -v tmp="$tmp" 'BEGIN{
    while ((getline < tmp) > 0) {
      tp[$1] = 1;
    }
    srand(1234);
  }{
    if (match($1, /^([0-9]+_[0-9]+_[0-9]+)\..*$/, a)) {
      if (a[1] in tp) {
        print $0 > "data/bentham/lang/char/te.txt";
      } else if (rand() < 0.1) {
        print $0 > "data/bentham/lang/char/va.txt";
      } else {
        print $0 > "data/bentham/lang/char/tr.txt";
      }
    }
  }' data/bentham/lang/char/normalized.txt;
}
