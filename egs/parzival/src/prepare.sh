#!/bin/bash
export LC_NUMERIC=C;
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/..";

[ ! -f "../utils/parse_options.inc.sh" ] &&
echo "Missing \"$SDIR/../../utils/parse_options.inc.sh\" file!" >&2 && exit 1;

# Check required files
for f in data/parzivaldb-v1.0/ground_truth/transcription.txt; do
  [ ! -s "$f" ] && echo "ERROR: Required file \"$f\" does not exist!" >&2 &&
  exit 1;
done;

# Obtain character-level transcriptions of ALL the dataset.
mkdir -p data/lang/char;
[[ "$overwrite" = false && -s data/lang/char/transcript.txt ]] ||
gawk '{
  x=$1; y=$2;
  gsub(/-/, " ", y);
  gsub(/\|/, " <space> ", y);
  print x, y;
}' data/parzivaldb-v1.0/ground_truth/transcription.txt \
> data/lang/char/transcript.txt || exit 1;

# Get list of characters
[[ "$overwrite" = false && -s data/lang/char/syms.txt ]] ||
cut -d\  -f2- data/lang/char/transcript.txt | tr \  \\n | sort -uV |
gawk 'BEGIN{
  N = 0;
  printf("%-12s %d\n", "<eps>", N++);
  printf("%-12s %d\n", "<ctc>", N++);
  printf("%-12s %d\n", "<space>", N++);
}$1 !~ /<space>/{
  printf("%-12s %d\n", $1, N++);
}' > data/lang/char/syms.txt || exit 1;

# Create files for each cross-validation set.
for p in train valid test; do
  # Transcript filess for each partition
  [[ "$overwrite" = false && -s "data/lang/char/${p:0:2}.txt" ]] ||
  gawk -v LF="data/parzivaldb-v1.0/sets1/$p.txt" \
    'BEGIN{ while((getline < LF) > 0) { L[$1]=1; } }($1 in L){ print; }' \
    "data/lang/char/transcript.txt" > "data/lang/char/${p:0:2}.txt" || exit 1;
done;

exit 0;
