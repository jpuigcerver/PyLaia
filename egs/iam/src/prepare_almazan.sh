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

# Prepare character-level stop words list.
[ -s data/almazan/lang/char/stopwords.txt ] ||
awk '{
  printf("%s", substr($1, 1, 1));
  for (i=2; i<=length($1); ++i) printf(" %s", substr($1, i, 1));
  printf("\n");
}' data/almazan/lang/word/stopwords.txt > data/almazan/lang/char/stopwords.txt;

# Prepare list of all stopword samples.
[ -s data/almazan/lang/word/all_stopwords.txt ] ||
awk 'BEGIN{
  while ((getline < "data/almazan/lang/word/stopwords.txt") > 0) {
    sw[$1] = 1;
  }
}sw[$2] == 1' data/almazan/lang/word/all.txt \
    > data/almazan/lang/word/all_stopwords.txt;

# Split into test, train and validation sets.
for p in char word; do
    for s in te tr va; do
	[ -s data/almazan/lang/$p/$s.txt ] ||
	awk -v SF=data/almazan/$s.txt 'BEGIN{
          while((getline < SF) > 0) KEEP[$1] = 1;
        }{
          split($1, A, "/");
          if (A[3] in KEEP) print;
        }' data/almazan/lang/$p/all.txt > data/almazan/lang/$p/$s.txt;
    done;
done;

for s in te va; do
  [ -s data/almazan/lang/word/${s}_no_singleton.txt ] ||
  awk -v SF=data/almazan/$s.txt '{
    if ($2 in INST) INST[$2] = INST[$2]" "$1;
    else INST[$2] = $1;
    COUNT[$2] = COUNT[$2] + 1;
  }END{
    for (w in INST) {
      if (COUNT[w] > 1) {
        n = split(INST[w], A, " ");
        for (i = 1; i <= n; ++i) { print A[i], w; }
      }
    }
  }' data/almazan/lang/word/$s.txt |
  sort > data/almazan/lang/word/${s}_no_singleton.txt;

  [ -s data/almazan/lang/char/${s}_no_singleton.txt ] ||
  join -1 1 \
       <(cut -d\  -f1 data/almazan/lang/word/${s}_no_singleton.txt) \
       <(sort data/almazan/lang/char/$s.txt) \
       > data/almazan/lang/char/${s}_no_singleton.txt;
done;
