#!/bin/bash
set -e;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# CD to the main directory for the experiment.
cd "${SDIR}/..";

mkdir -p data/original/word_kws/lang/{char,word};

# Prepare word-level transcriptions.
[ -s data/original/word_kws/lang/word/all.txt ] ||
awk '$1 !~ /^#/{
  # Some words in the IAM dataset have a space in-between.
  word = $9;
  for (i=10; i<=NF; ++i) {
    word = sprintf("%s%s", word, $i);
  }
  # Consider only lowercase words, as Almazan does.
  word = tolower(word);
  # Remove other symbols than [0-9a-z], as Almazan does.
  gsub(/[^0-9a-z]+/, "", word);
  # Words that were punctuation symbols are simply replaced with "-",
  # as Almazan does.
  if (word == "") word = "-";

  split($1, arr, "-");
  printf("%-27s %s\n", arr[1]"/"arr[1]"-"arr[2]"/"$1, word);
}' data/original/ascii/words.txt > data/original/word_kws/lang/word/all.txt;

# Prepare char-level transcriptions.
[ -s data/original/word_kws/lang/char/all.txt ] ||
awk '{
  printf("%-27s", $1);
  for (i=1;i<=length($2);++i) {
    printf(" %s", substr($2, i, 1));
  }
  printf("\n");
}' data/original/word_kws/lang/word/all.txt \
  > data/original/word_kws/lang/char/all.txt;

# Prepare symbols list for PHOCNet training
[ -s data/original/word_kws/lang/syms_phoc.txt ] ||
cut -d\  -f2- data/original/word_kws/lang/char/all.txt | tr \  \\n |
awk 'NF > 0' | sort -uV | awk '{
  printf("%s %d\n", $1, NR - 1);
}' > data/original/word_kws/lang/syms_phoc.txt;

# Prepare symbols list for CTCNet training
[ -s data/original/word_kws/lang/syms_ctc.txt ] ||
awk 'BEGIN{
  printf("<ctc> %d\n", NR);
}{
  printf("%-5s %d\n", $1, NR);
}' data/original/word_kws/lang/syms_phoc.txt \
  > data/original/word_kws/lang/syms_ctc.txt;

# Prepare stop words list.
[ -s data/original/word_kws/lang/word/stopwords.txt ] ||
cat data/almazan/swIAM.txt |
tr \, \\n > data/original/word_kws/lang/word/stopwords.txt;

# Prepare character-level stop words list.
[ -s data/original/word_kws/lang/char/stopwords.txt ] ||
awk '{
  printf("%s", substr($1, 1, 1));
  for (i=2; i<=length($1); ++i) printf(" %s", substr($1, i, 1));
  printf("\n");
}' data/original/word_kws/lang/word/stopwords.txt \
  > data/original/word_kws/lang/char/stopwords.txt;

# Prepare list of all stopword samples.
[ -s data/original/word_kws/lang/word/all_stopwords.txt ] ||
awk 'BEGIN{
  while ((getline < "data/original/word_kws/lang/word/stopwords.txt") > 0) {
    sw[$1] = 1;
  }
}sw[$2] == 1' data/original/word_kws/lang/word/all.txt \
    > data/original/word_kws/lang/word/all_stopwords.txt;

# Split into test, train and validation sets.
for p in char word; do
    for s in te tr va; do
        [ -s data/original/word_kws/lang/$p/$s.txt ] && continue;
        if [ "$s" = te ]; then
            partfile=data/original/testset.txt;
        elif [ "$s" = tr ]; then
            partfile=data/original/trainset.txt;
        else
            partfile="$(mktemp)";
            cat data/original/validationset?.txt > "$partfile";
        fi;
	    awk -v SF="$partfile" 'BEGIN{
          while((getline < SF) > 0) KEEP[$1] = 1;
        }{
          split($1, A, "/");
          split(A[3], A, "-");
          if (A[1]"-"A[2]"-"A[3] in KEEP) print;
        }' data/original/word_kws/lang/$p/all.txt \
          > data/original/word_kws/lang/$p/$s.txt;
    done;
done;

for s in te va; do
  [ -s data/original/word_kws/lang/word/${s}_no_singleton.txt ] ||
  awk -v SF=data/original/word_kws/$s.txt '{
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
  }' data/original/word_kws/lang/word/$s.txt |
  sort > data/original/word_kws/lang/word/${s}_no_singleton.txt;

  [ -s data/original/word_kws/lang/char/${s}_no_singleton.txt ] ||
  join -1 1 \
       <(cut -d\  -f1 data/original/word_kws/lang/word/${s}_no_singleton.txt) \
       <(sort data/original/word_kws/lang/char/$s.txt) \
       > data/original/word_kws/lang/char/${s}_no_singleton.txt;
done;
