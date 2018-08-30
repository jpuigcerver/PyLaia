#!/bin/bash
set -e;
export LC_NUMERIC=C;

wspace="<space>";
mkdir -p data/kws_line/lang/{char,word};

# Prepare word-level transcripts.
[ -s "data/kws_line/lang/word/all.txt" ] ||
gawk '$1 !~ /^#/' "data/original/ascii/lines.txt" | cut -d\  -f1,9- |
gawk '{ $1=$1"|"; print; }' |
# Some words include spaces (e.g. "B B C" -> "BBC"), remove them.
sed -r 's| +||g' |
# Replace character | with whitespaces.
tr \| \  |
# Some contractions where separated from the words to reduce the vocabulary
# size. These separations are unreal, we join them (e.g. "We 'll" -> "We'll").
sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' |
sort -k1 > "data/kws_line/lang/word/all.txt" ||
{ echo "ERROR: Creating file data/kws_line/lang/word/all.txt" >&2 && exit 1; }

# Prepare character-level transcripts.
[ -s "data/kws_line/lang/char/all.txt" ] ||
gawk -v ws="$wspace" '{
  printf("%s", $1);
  for(i=2;i<=NF;++i) {
    for(j=1;j<=length($i);++j) {
      printf(" %s", substr($i, j, 1));
    }
    if (i < NF) printf(" %s", ws);
  }
  printf("\n");
}' "data/kws_line/lang/word/all.txt" |
sort -k1 > "data/kws_line/lang/char/all.txt" ||
{ echo "ERROR: Creating file data/kws_line/lang/char/all.txt" >&2 && exit 1; }

# Download KWS partition
[ -s data/kws_line/lang/word/te.txt -a \
  -s data/kws_line/lang/word/tr.txt -a \
  -s data/kws_line/lang/word/va.txt ] || {
  tmpd="$(mktemp -d)";
  wget --no-check-certificate -P "$tmpd" -N \
       https://www.prhlt.upv.es/~jpuigcerver/iam_kws_partition.tar.gz
  tar zxf "$tmpd/iam_kws_partition.tar.gz" -C "$tmpd";
  for p in te tr va; do
    join -1 1 <(sort -k1 "$tmpd/$p.lst") data/kws_line/lang/word/all.txt \
	 > "data/kws_line/lang/word/$p.txt";
  done;
  rm -rf "$tmpd";
}

for p in te tr va; do
  [ -s "data/kws_line/lang/char/$p.txt" ] ||
  join -1 1 \
       <(cut -d\  -f1 "data/kws_line/lang/word/$p.txt") \
       data/kws_line/lang/char/all.txt > "data/kws_line/lang/char/$p.txt";
done;

# Prepare list of training CTC symbols
[ -s data/kws_line/lang/char/syms_ctc.txt ] ||
cut -d\  -f2- data/kws_line/lang/char/{tr,va}.txt |
tr \  \\n | sort -uV |
gawk 'BEGIN{
  N=0;
  printf("%5-s %d\n", "<ctc>", N++);
}{
  printf("%5-s %d\n", $1, N++);
}' > data/kws_line/lang/char/syms_ctc.txt;

# Prepare lowercase-only transcriptions.
for p in all te tr va; do
  for t in char word; do
    [ -s "data/kws_line/lang/$t/${p}_lowercase.txt" ] ||
    gawk '{ for (i=1; i<=NF; ++i) { $i=tolower($i); }; print }' \
	"data/kws_line/lang/$t/$p.txt" \
	> "data/kws_line/lang/$t/${p}_lowercase.txt";
  done;
done;

# Prepare list of training CTC symbols
[ -s data/kws_line/lang/char/syms_ctc_lowercase.txt ] ||
cut -d\  -f2- data/kws_line/lang/char/{tr,va}_lowercase.txt |
tr \  \\n | sort -uV |
gawk 'BEGIN{
  N=0;
  printf("%5-s %d\n", "<ctc>", N++);
}{
  printf("%5-s %d\n", $1, N++);
}' > data/kws_line/lang/char/syms_ctc_lowercase.txt;

# Prepare KWS reference files
mkdir -p data/kws_line/lang/kws_refs;
for p in te te_lowercase tr tr_lowercase va va_lowercase; do
  [ -s "data/kws_line/lang/kws_refs/$p.txt" ] ||
  awk '{for (i=2;i<=NF;++i) print $i, $1; }' "data/kws_line/lang/word/$p.txt" |
  sort -u > "data/kws_line/lang/kws_refs/$p.txt" || exit 1;
done;

# Download full list of English stopwords
[ -s data/kws_line/lang/english_stopwords.txt ] ||
wget -P data/kws_line/lang -N \
   https://gist.githubusercontent.com/jpuigcerver/c0c5bc715f2df2bfde8d0d79ae049a05/raw/926fa0c1379eec2d562a97875ffdc55aa39f7ca6/english_stopwords.txt || exit 1;

# Prepare query lists
mkdir -p data/kws_line/lang/queries;
for p in te tr va; do
  # Case-insensitive queries
  [ -s data/kws_line/lang/queries/${p}_lowercase.txt ] ||
  cut -d\  -f1 data/kws_line/lang/kws_refs/${p}_lowercase.txt | sort -u |
  comm -23 - <(sort data/kws_line/lang/english_stopwords.txt) |
  gawk '$0 ~ /[0-9a-z]+/' |
  sort > data/kws_line/lang/queries/${p}_lowercase.txt || exit 1;
  # Case-sensitive queries
  [ -s data/kws_line/lang/queries/${p}.txt ] ||
  cut -d\  -f1 data/kws_line/lang/kws_refs/${p}.txt | sort -u |
  gawk -v qfile=data/kws_line/lang/queries/${p}_lowercase.txt \
  'BEGIN{ while((getline < qfile) > 0) Q[$1] = 1; }(tolower($0) in Q)' |
  sort > data/kws_line/lang/queries/${p}.txt || exit 1;
done;

# Download list of queries used in old KWS experiments for IAM
[ -s data/kws_line/lang/queries/iam_old_papers_queries.txt ] ||
wget -P data/kws_line/lang/queries -N \
     https://gist.githubusercontent.com/jpuigcerver/a2c94a2196211aea6b2c08774f50f541/raw/de52fe19e53ecb87df8e3e614e4e5815c4a045d9/iam_old_papers_queries.txt || exit 1;

# Convert old queries to lowercase
[ -s data/kws_line/lang/queries/iam_old_papers_queries_lowercase.txt ] ||
awk '{ print tolower($0); }' \
  data/kws_line/lang/queries/iam_old_papers_queries.txt |
sort -u > \
  data/kws_line/lang/queries/iam_old_papers_queries_lowercase.txt || exit 1;


function process_image_resize_h128 () {
  local bn="$(basename "$1" .png)";
  [ -s "data/kws_line/imgs/resize_h128/$bn.png" ] ||
  imgtxtenh -d 118.110 "$1" png:- |
  convert png:- -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    -resize "x128" +repage -strip \
    "data/kws_line/imgs/resize_h128/$bn.png" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  return 0;
}

function process_image_original_size () {
  local bn="$(basename "$1" .png)";
  [ -s "data/kws_line/imgs/original_size/$bn.png" ] ||
  imgtxtenh -d 118.110 "$1" png:- |
  convert png:- -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage -strip \
    "data/kws_line/imgs/original_size/$bn.png" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  return 0;
}

function process_image_pad_h128 () {
  local bn="$(basename "$1" .png)";
  tmpf="$(mktemp)";
  [ -s "data/kws_line/imgs/padding_h128/$bn.png" ] && return 0;
  imgtxtenh -d 118.110 "$1" png:- |
  convert png:- -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    "$tmpf.png" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  h="$(identify -format '%h' "$tmpf.png")";
  if [ "$h" -lt 128 ]; then
    convert -gravity center -extent "x128" +repage -strip \
	    "$tmpf.png" "data/kws_line/imgs/padding_h128/$bn.png";
  else
    convert -resize "x128" +repage -strip \
	    "$tmpf.png" "data/kws_line/imgs/padding_h128/$bn.png";
  fi ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  rm -f "$tmpf" "$tmpf.png";
  return 0;
}

mkdir -p data/kws_line/imgs/{padding_h128,resize_h128,original_size};
readarray -t original_imgs < <(find data/original/lines -name "*.png");

[ "$(find data/kws_line/imgs/resize_h128 -name "*.png" | wc -l)" -eq \
  "${#original_imgs[@]}" ] || {
  n=0;
  for f in "${original_imgs[@]}"; do
    ((++n));
    process_image_resize_h128 "$f" &
    if [ "$n" -eq "$(nproc)" ]; then wait || exit 1; n=0; fi;
  done;
  wait || exit 1;
}

[ "$(find data/kws_line/imgs/padding_h128 -name "*.png" | wc -l)" -eq \
  "${#original_imgs[@]}" ] || {
  n=0;
  for f in "${original_imgs[@]}"; do
    ((++n));
    process_image_pad_h128 "$f" &
    if [ "$n" -eq "$(nproc)" ]; then wait || exit 1; n=0; fi;
  done;
  wait || exit 1;
}

[ "$(find data/kws_line/imgs/original_size -name "*.png" | wc -l)" -eq \
  "${#original_imgs[@]}" ] || {
  n=0;
  for f in "${original_imgs[@]}"; do
    ((++n));
    process_image_original_size "$f" &
    if [ "$n" -eq "$(nproc)" ]; then wait || exit 1; n=0; fi;
  done;
  wait || exit 1;
}

# List of delimiter characters used by lexicon-free KWS methods
[ -s data/kws_line/lang/delimiters.txt ] ||
cat <<EOF > data/kws_line/lang/delimiters.txt
<ctc>
#
&
(
)
*
:
;
<space>
?
<unk>
EOF

# Use all available training lines (11504).
[ -s data/kws_line/lang/char/tr.11504.txt ] ||
join -1 1 \
  <(sort -k1b,1 data/kws_line/lang/char/all.txt) \
  <(comm -23 <(cut -d\  -f1 data/kws_line/lang/char/all.txt | sort -k1b,1) \
             <(cat data/kws_line/lang/char/{te,va}.txt | cut -d\  -f1 | sort -k1b,1)) |
awk 'BEGIN{
  while ((getline < "data/kws_line/lang/char/syms_ctc.txt") > 0) {
    SYMS[$1] = $2;
  }
}NF > 1{
  ok = 1;
  for (i = 2; i <= NF && ok; ++i) {
    if (!($i in SYMS)) ok = 0;
  }
  if (ok) print;
}' > data/kws_line/lang/char/tr.11504.txt;

# Select smaller subsets of all the available training data
# (1/3, 2/3, 4/3 of the original training set).
for n in 2054 4108 8216; do
    [ -s "data/kws_line/lang/char/tr.$n.txt" ] ||
    sort -R --random-source=data/original/ascii/lines.txt \
        data/kws_line/lang/char/tr.11504.txt |
    head -n "$n" |
    sort > "data/kws_line/lang/char/tr.$n.txt";
done;
