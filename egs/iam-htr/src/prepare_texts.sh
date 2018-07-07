#!/bin/bash
set -e;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/src" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/src/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/src/parse_options.inc.sh file!" >&2 && exit 1;

overwrite=false;
wspace="<space>";
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
  --wspace     : (type = string, default = \"$wspace\")
                 Use this symbol to represent the whitespace character.
";
source "$(pwd)/src/parse_options.inc.sh" || exit 1;

mkdir -p data/lists/lines/{aachen,original};

# Prepare image lists
for c in aachen original; do
  for f in data/part/lines/$c/*.lst; do
    bn=$(basename "$f" .lst);
    [ -s "data/lists/lines/$c/$bn.lst" ] ||
    gawk '{ print $1 }' "$f" \
    > "data/lists/lines/$c/$bn.lst" || exit 1;
  done;
done;

mkdir -p data/lang/{lines,forms}/{char,word}/aachen;

# Prepare word-level transcripts.
[ "$overwrite" = false -a -s "data/lang/lines/word/all.txt" ] ||
gawk '$1 !~ /^#/' "data/original/lines.txt" | cut -d\  -f1,9- |
gawk '{ $1=$1"|"; print; }' |
# Some words include spaces (e.g. "B B C" -> "BBC"), remove them.
sed -r 's| +||g' |
# Replace character | with whitespaces.
tr \| \  |
# Some contractions where separated from the words to reduce the vocabulary
# size. These separations are unreal, we join them (e.g. "We 'll" -> "We'll").
sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' |
sort -k1 > "data/lang/lines/word/all.txt" ||
{ echo "ERROR: Creating file data/lang/lines/word/all.txt" >&2 && exit 1; }

# Prepare character-level transcripts.
[ "$overwrite" = false -a -s "data/lang/lines/char/all.txt" ] ||
gawk -v ws="$wspace" '{
  printf("%s", $1);
  for(i=2;i<=NF;++i) {
    for(j=1;j<=length($i);++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" %s", ws);
  }
  printf("\n");
}' "data/lang/lines/word/all.txt" |
sort -k1 > "data/lang/lines/char/all.txt" ||
{ echo "ERROR: Creating file data/lang/lines/char/all.txt" >&2 && exit 1; }

# Extract characters list for training.
mkdir -p "train";
[ "$overwrite" = false -a -s "train/syms.txt" ] ||
cut -d\  -f2- "data/lang/lines/char/all.txt" | tr \  \\n | sort | uniq |
gawk -v ws="$wspace" 'BEGIN{
  printf("%-12s %d\n", "<ctc>", 0);
  printf("%-12s %d\n", ws, 1);
  N = 2;
}$1 != ws{
  printf("%-12s %d\n", $1, N++);
}' > "train/syms.txt" ||
{ echo "ERROR: Creating file train/syms.txt" >&2 && exit 1; }

# Split files into different partitions (train, test, valid).
for p in aachen/{tr,te,va}; do
  join -1 1 "data/part/lines/$p.lst" "data/lang/lines/char/all.txt" \
    > "data/lang/lines/char/$p.txt" ||
  { echo "ERROR: Creating file data/lang/lines/char/$p.txt" >&2 && exit 1; }
  join -1 1 "data/part/lines/$p.lst" "data/lang/lines/word/all.txt" \
    > "data/lang/lines/word/$p.txt" ||
  { echo "ERROR: Creating file data/lang/lines/word/$p.txt" >&2 && exit 1; }
done;

for p in tr te va; do
  txtw="data/lang/forms/word/aachen/$p.txt";
  txtc="data/lang/forms/char/aachen/$p.txt";
  # Get the word-level transcript of the whole form.
  [[ "$overwrite" = false && -s "$txtw" &&
      ( ! "$txtw" -ot "data/lang/lines/word/aachen/$p.txt" ) ]] ||
  gawk 'BEGIN{ sent_id=""; }{
    if (match($0, /^([^ ]+)-[0-9]+ (.+)$/, A)) {
      if (A[1] != sent_id) {
        if (sent_id != "") printf("\n");
        printf("%s %s", A[1], A[2]);
        sent_id = A[1];
      } else {
        printf(" %s", A[2]);
      }
    }
  }END{ if (sent_id != "") printf("\n"); }' \
    "data/lang/lines/word/aachen/$p.txt" > "$txtw" ||
  { echo "ERROR: Creating file \"$txtw\"!" >&2 && exit 1; }
  # Prepare character-level transcripts.
  [ "$overwrite" = false -a -s "$txtc" ] ||
  gawk -v ws="$wspace" '{
    printf("%s", $1);
    for(i=2;i<=NF;++i) {
      for(j=1;j<=length($i);++j) {
        printf(" %s", substr($i, j, 1));
      }
      if (i < NF) printf(" %s", ws);
    }
    printf("\n");
  }' "$txtw" > "$txtc" ||
  { echo "ERROR: Creating file $txtc" >&2 && exit 1; }
done;

exit 0;
