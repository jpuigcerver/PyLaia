#!/bin/bash
export LC_NUMERIC=C;
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/..";

overwrite=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite  : (type = boolean, default = $overwrite)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

# Check required files
for f in data/parzivaldb-v1.0/ground_truth/transcription.txt; do
  [ ! -s "$f" ] && echo "ERROR: Required file \"$f\" does not exist!" >&2 &&
  exit 1;
done;

# Obtain character-level transcripts of ALL the dataset.
mkdir -p data/lang/char;
[[ "$overwrite" = false && -s data/lang/char/transcript.txt ]] ||
gawk '{
  x=$1; y=$2;
  gsub(/-/, " ", y);
  gsub(/\|/, " <space> ", y);
  print x, y;
}' data/parzivaldb-v1.0/ground_truth/transcription.txt \
> data/lang/char/transcript.txt || exit 1;

# Obtain word-level transcripts
mkdir -p data/lang/word;
[[ "$overwrite" = false && -s data/lang/word/transcript.txt ]] ||
sed -r 's/[|]/ /g' data/parzivaldb-v1.0/ground_truth/transcription.txt \
> data/lang/word/transcript.txt || exit 1;

# Get list of characters
[[ "$overwrite" = false && -s data/lang/char/syms_ctc.txt ]] ||
cut -d\  -f2- data/lang/char/transcript.txt | tr \  \\n | sort -uV |
gawk 'BEGIN{
  N = 0;
  printf("%-12s %d\n", "<ctc>", N++);
  printf("%-12s %d\n", "<space>", N++);
}$1 !~ /<space>/{
  printf("%-12s %d\n", $1, N++);
}' > data/lang/char/syms_ctc.txt || exit 1;

# Create transcript files for each partition.
for p in train valid test; do
  for c in char word; do
    [[ "$overwrite" = false && -s "data/lang/$c/${p:0:2}.txt" ]] ||
    join -1 1 <(sort -k1b,1 "data/parzivaldb-v1.0/sets1/$p.txt") \
              <(sort -k1b,1 "data/lang/$c/transcript.txt") \
              > "data/lang/$c/${p:0:2}.txt" || exit 1;
  done;
done;

# Create KWS reference file for ALL partitions.
mkdir -p data/lang/kws_refs;
[[ "$overwrite" = false && -s data/lang/kws_refs/all.txt ]] ||
gawk '{
  for (i = 2; i <= NF; ++i) {
    print $i, $1;
  }
}' data/lang/word/transcript.txt |
sort -u > data/lang/kws_refs/all.txt || exit 1;

# Split KWS reference files for each partition (train, test, valid).
for p in train valid test; do
  [[ "$overwrite" = false && -s "data/lang/kws_refs/${p:0:2}.txt" ]] ||
  join -1 1 -2 2 \
      <(sort -k1b,1 "data/parzivaldb-v1.0/sets1/$p.txt") \
      <(sort -k2b,2 "data/lang/kws_refs/all.txt") |
      awk '{ print $2, $1}' |
      sort > "data/lang/kws_refs/${p:0:2}.txt" || exit 1;
done;

[[ "$overwrite" = false && -s data/lang/delimiters.txt ]] ||
cat <<EOF > data/lang/delimiters.txt
<space>
eq
pt
EOF

exit 0;
