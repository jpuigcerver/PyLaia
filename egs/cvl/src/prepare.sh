#!/bin/bash
set -e;
export LC_ALL=en_US.utf-8;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

mkdir -p data/images/words;
[ "$(find data/images/words -name "*.png" | wc -l)" -eq 99904 ] || {
  tmp="$(mktemp)";
  find data/cvl-database-1-1/{testset,trainset}/words -name "*.tif" > "$tmp";
  [ "$(wc -l "$tmp" | cut -d\  -f1)" -ne 99904 ] &&
  echo "Missing original TIF files. Did you download the dataset?" >&2 && exit 1;
  split -n"l/$(nproc)" "$tmp" "$tmp.p";
  readarray -t tmpa <  <(ls "$tmp.p"*);
  for f in "${tmpa[@]}"; do
    gawk '{
      n=split($1, A, "/");
      bn=substr(A[n], 1, length(A[n]) - 4);
      cmd=sprintf("convert \"%s\" \"data/images/words/%s.png\" 2>&1", $1, bn);
      cmd | getline cmd_output;
      status = close(cmd);
      if (status != 0 && match(cmd_output, /^convert: TIFF directory is missing required "ImageLength" field.+$/) == 0) {
         print "Error processing image "$1"\n"cmd_output > "/dev/stderr";
      }
    }' "$f" &
  done;
  wait;
}

mkdir -p data/kws/lang/words/word;
[ -s data/kws/lang/words/word/all.txt ] ||
find data/images/words -name "*.png" |
awk '{
  match($1, /^.+\/(.+)\.png$/, A);
  bn = A[1];
  match(bn, /^[0-9]+-[0-9]+-[0-9]+-[0-9]+-(.+)$/, A);
  A[1] = gensub(/\#U(....)/, "<U\\1>", "g", A[1]);
  print bn, A[1];
}' |
sort -V | ascii2uni -a A -q |
gawk '{
  gsub(/'\''/, "", $2);
  $2 = toupper($2);
  print;
}' > data/kws/lang/words/word/all.txt;

[ -s data/kws/lang/syms_phoc.txt ] ||
cut -d\  -f2 data/kws/lang/words/word/all.txt |
awk '{ for(i=1; i<=length($1); ++i) print substr($1, i, 1); }' |
sort -uV | awk '{ print $1, NR - 1; }' > data/kws/lang/syms_phoc.txt;

[ -s data/kws/lang/syms_ctc.txt ] ||
awk 'BEGIN{ print "<ctc>", 0; }{ print $1, NR; }' data/kws/lang/syms_phoc.txt \
    > data/kws/lang/syms_ctc.txt;

[ -s data/kws/lang/words/word/tr.txt ] ||
head -n12147 data/kws/lang/words/word/all.txt > data/kws/lang/words/word/tr.txt;

[ -s data/kws/lang/words/word/te.txt ] ||
comm -23 <(sort data/kws/lang/words/word/all.txt) <(sort data/kws/lang/words/word/tr.txt) |
sort -V > data/kws/lang/words/word/te.txt;

# Select 1000 random test queries.
# These will be matched against all test word images.
[ -s data/kws/lang/words/word/te_queries.txt ] ||
awk 'length($2) > 3' data/kws/lang/words/word/te.txt |
sort --random-source=data/kws/lang/words/word/te.txt -R |
head -n1000 | sort -V > data/kws/lang/words/word/te_queries.txt;

# Create a small validation set from the training set.
[ -s data/kws/lang/words/word/tr.2.txt ] ||
awk 'length($2) > 3' data/kws/lang/words/word/tr.txt |
sort --random-source=data/kws/lang/words/word/tr.txt -R |
head -n750 | sort -V > data/kws/lang/words/word/tr.2.txt;

# Remove the selected validation images from the training.
[ -s data/kws/lang/words/word/tr.1.txt ] ||
comm -23 <(sort data/kws/lang/words/word/tr.txt) <(sort data/kws/lang/words/word/tr.2.txt) |
sort -V > data/kws/lang/words/word/tr.1.txt;

mkdir -p data/kws/lang/words/char;
for f in data/kws/lang/words/word/tr.1.txt data/kws/lang/words/word/tr.2.txt; do
  out="${f/\/word\//\/char\/}";
  [ -s "$out" ] ||
  awk '{
    printf("%s", $1);
    for(i=1; i<=length($2); ++i) {
      printf(" %s", substr($2, i, 1));
    }
    printf("\n");
  }' "$f" > "$out";
done;
