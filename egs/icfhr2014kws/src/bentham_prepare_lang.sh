#!/bin/bash
set -e;
export LC_ALL=en_US.utf8;
export LC_NUMERIC=C;

# Check for necessary files.
for d in data/duth/TRACK_II_Bentham_Dataset \
	 data/prhlt/contestHTRtS/BenthamData/Transcriptions; do
  [ ! -d "$d" ] && echo "Directory \"$d\" does not exist!" >&2 && exit 1;
done;

mkdir -p data/bentham/lang/{char,word};
[ -s data/bentham/lang/word/original.txt ] ||
find data/prhlt/contestHTRtS/BenthamData/Transcriptions -name "*.txt" |
xargs awk '{
  if (NF == 0) next;
  match(FILENAME, /^.*\/([0-9_]+).txt/, arr);
  print arr[1], $0;
}' | sort | sed -r 's| +| |g;s| $||g' > data/bentham/lang/word/original.txt;

# First, put all words in lowercase.
# Then, place suffixes like 'd 's 'll 've to next to the previous word
# Next, remove ' and ", since these are not relevant for KWS
# Finally, replace delimiters , . : ;  with whitespace
[ -s data/bentham/lang/word/normalized.txt ] ||
awk '{
  printf("%s", $1);
  for (i = 2; i <= NF; ++i) { printf(" %s", tolower($i));  }
  printf("\n");
}' data/bentham/lang/word/original.txt |
sed -r 's/[ ]+'\''[ ]?(s|d|ll|ve) /'\''\1 /g' |
sed -r 's|['\''"]||g' |
sed -r 's|[,.:;]| |g' |
sed -r 's| +| |g;s| $||g' > data/bentham/lang/word/normalized.txt;

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
    if (match($1, /^([0-9]+_[0-9]+_[0-9]+)_.*$/, a)) {
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
