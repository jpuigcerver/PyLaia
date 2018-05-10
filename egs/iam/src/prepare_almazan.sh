#!/bin/bash
set -e;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# CD to the main directory for the experiment.
cd "${SDIR}/..";

mkdir -p data/almazan/lang/{char,word};
# Prepare word-level transcriptions.
[ -s data/almazan/lang/word/all.txt ] ||
awk '{
  img = $NF;
  wrd = tolower($(NF - 1));
  split(img, arr, "-");
  printf("%-27s %s\n", arr[1]"/"arr[1]"-"arr[2]"/"img, wrd);
}' data/almazan/queries.gtp > data/almazan/lang/word/all.txt;

# Prepare char-level transcriptions.
[ -s data/almazan/lang/char/all.txt ] ||
awk '{
  printf("%-27s", $1);
  for (i=1;i<=length($2);++i) {
    printf(" %s", substr($2, i, 1));
  }
  printf("\n");
}' data/almazan/lang/word/all.txt > data/almazan/lang/char/all.txt;

# Prepare symbols list for PHOCNet training
[ -s data/almazan/lang/syms_phoc.txt ] ||
cut -d\  -f2- data/almazan/lang/char/all.txt | tr \  \\n | awk 'NF > 0' |
sort -uV | awk '{
  printf("%s %d\n", $1, NR - 1);
}' > data/almazan/lang/syms_phoc.txt;

# Prepare symbols list for CTCNet training
[ -s data/almazan/lang/syms_ctc.txt ] ||
awk 'BEGIN{
  printf("<ctc> %d\n", NR);
}{
  printf("%-5s %d\n", $1, NR);
}' data/almazan/lang/syms_phoc.txt > data/almazan/lang/syms_ctc.txt;

# Prepare stop words list.
[ -s data/almazan/lang/word/stopwords.txt ] ||
  cat data/almazan/swIAM.txt | tr \, \\n > data/almazan/lang/word/stopwords.txt;

# Split into test, train and validation sets.
for p in char word; do
  for s in te tr va; do
    [ -s data/almazan/lang/$p/$s.txt ] ||
      join -1 1 \
	   <(sort data/almazan/lang/$p/all.txt) \
	   <(awk '{ split($1, A, "-"); print A[1]"/"A[1]"-"A[2]"/"$1; }' \
		 data/almazan/$s.txt | sort) \
	   > data/almazan/lang/$p/$s.txt;
  done;
done;

for s in te va; do
  # Get counts of the validation and test partitions
  [ -s data/almazan/lang/word/$s.voc ] ||
  cut -d\  -f2- data/almazan/lang/word/$s.txt |
  sort | uniq -c > data/almazan/lang/word/$s.voc;
  # Get distractor samples (stopwords + words with count < 2)
  [ -s data/almazan/lang/word/${s}_distractors.txt ] || {
    join -2 2 \
    	 <(sort data/almazan/lang/word/stopwords.txt) \
    	 <(sort -k2 data/almazan/lang/word/$s.txt);
    join -1 2 \
    	 <(sort -k2 data/almazan/lang/word/$s.txt) \
    	 <(awk '$1 < 2{print $2}' data/almazan/lang/word/$s.voc | sort);
  } | awk '{ print $2, $1; }' |
  sort -u > data/almazan/lang/word/${s}_distractors.txt;
  # Get query samples (the rest)
  [ -s data/almazan/lang/word/${s}_queries.txt ] ||
  comm -13 \
       data/almazan/lang/word/${s}_distractors.txt \
       data/almazan/lang/word/${s}.txt \
       > data/almazan/lang/word/${s}_queries.txt;
  # Get character-level transcription of the same files.
  for f in distractors queries; do
    inp="data/almazan/lang/word/${s}_${f}.txt";
    out="data/almazan/lang/char/${s}_${f}.txt";
    [ -s "$out" ] ||
    join -1 1 \
	 <(cut -d\  -f1 "$inp") \
	 data/almazan/lang/char/${s}.txt \
	 > "$out";
  done;

  [ -s "data/almazan/${s}_qbe_rel_pairs.txt" ] ||
  awk '{
    if ($2 in IMGS) {
      IMGS[$2] = IMGS[$2]" "$1;
    } else {
      IMGS[$2] = $1;
    }
  }END{
    for (w in IMGS) {
      n = split(IMGS[w], A, " ");
      for (i = 1; i <= n; ++i) {
        for (j = i + 1; j <= n; ++j) {
          print A[i], A[j];
          print A[j], A[i];
        }
      }
    }
  }' "data/almazan/lang/word/${s}_queries.txt" \
      > "data/almazan/${s}_qbe_rel_pairs.txt";
done;
