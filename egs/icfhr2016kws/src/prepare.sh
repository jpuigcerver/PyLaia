#!/bin/bash
set -e;
export LC_ALL=en_US.utf8;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

mkdir -p data/images/word;
DATASETS=(Botany Konzilsprotokolle);
WPREFIX=(bt kt);
for d in 0 1; do
  ds=${DATASETS[d]};
  mkdir -p data/lang/${ds}/words/{char,word};
  [ -s data/lang/${ds}/words/word/tr+va.3.txt ] || {
    tmp=$(mktemp);
    for s in I II III; do
      awk '$1 == "<spot"' data/xmls/${ds}_Train_${s}_WL.xml
    done |
    sed -r 's|^.* word="([^"]+)" image="([^"]+)" x="([^"]+)" y="([^"]+)" w="([^"]+)" h="([^"]+)".*$|\1 \2 \3 \4 \5 \6|g' |
    awk -v wp=${WPREFIX[d]} '{
      printf("%s%06d %s %s %d %d %d %d\n", wp, NR, $1, $2, $3, $4, $5, $6);
    }' > "$tmp";
    split "-nl/$(nproc)" "$tmp" "$tmp.p";
    readarray -t tmpa <<< "$(ls "$tmp.p"*)";
    for t in "${tmpa[@]}"; do
      awk '{
        cmd=sprintf("convert data/images/page/%s -crop %dx%d+%d+%d +repage -strip data/images/word/%s.jpg",
                $3, $6, $7, $4, $5, $1);
        system(cmd);
      }' "$t"
    done;
    wait;
    awk '{ print $1, $2 }' "$tmp" |
    sed -r 's|&amp\;|\&|g;s|&quot\;|"|g' \
	> "data/lang/${ds}/words/word/tr+va.3.txt";
  }
  [ -s data/lang/${ds}/words/word/tr+va_ci.3.txt ] || {
    tmp=$(mktemp);
    for s in I II III; do
      awk '$1 == "<spot"' data/xmls/${ds}_Train_${s}_WL_CASE_INSENSITIVE.xml
    done |
    sed -r 's|^.* word="([^"]+)" .*$|\1|g' |
    awk -v wp=${WPREFIX[d]} '{
      printf("%s%06d %s\n", wp, NR, toupper($1));
    }' |
    sed -r 's|&AMP\;|\&|g;s|&QUOT\;|"|g' \
	> "data/lang/${ds}/words/word/tr+va_ci.3.txt";
  }
  [ -s data/lang/${ds}/words/char/tr+va_ci.3.txt ] || {
    awk '{
      printf("%s", $1);
      for (i=1; i <= length($2); ++i) {
        printf(" %s", substr($2, i, 1));
      }
      printf("\n");
    }' "data/lang/${ds}/words/word/tr+va_ci.3.txt" \
    > "data/lang/${ds}/words/char/tr+va_ci.3.txt";
  }
  [ -s data/lang/${ds}/syms_ci_phoc.txt ] ||
  cut -d\  -f2 "data/lang/${ds}/words/word/tr+va_ci.3.txt" |
  awk '{ for(i=1;i<=length($0);++i) print substr($0, i, 1); }' |
  sort -uV |
  awk '{print $1, NR - 1; }' > "data/lang/${ds}/syms_ci_phoc.txt";
done;

head -n1684 data/lang/Botany/words/word/tr+va.3.txt \
     > data/lang/Botany/words/word/tr+va.1.txt;
head -n5295 data/lang/Botany/words/word/tr+va.3.txt \
     > data/lang/Botany/words/word/tr+va.2.txt;

head -n1849 data/lang/Konzilsprotokolle/words/word/tr+va.3.txt \
     > data/lang/Konzilsprotokolle/words/word/tr+va.1.txt;
head -n7817 data/lang/Konzilsprotokolle/words/word/tr+va.3.txt \
     > data/lang/Konzilsprotokolle/words/word/tr+va.2.txt;

WPREFIX=(bw kw);
for d in 0 1; do
  ds=${DATASETS[d]};
  [ -s data/lang/$ds/words/word/te_words.txt ] || {
    tmp=$(mktemp);
    awk '$1 == "<spot"' "data/xmls/${ds}_Test_GT_SegBased_QbS.xml" |
    sed -r 's|^.* word="([^"]+)" image="([^"]+).jpg" .*$|\2 \1|g' > "$tmp";
    find data/images/word -name "${WPREFIX[d]}*" |
    sed -r 's|^.*\/([a-z0-9]+)\.jpg$|\1|g' | sort |
    awk -v TMPF="$tmp" 'BEGIN{
      while((getline < TMPF) > 0) { TXT[$1]=$2; }
    }{
      if ($1 in TXT) print $1, TXT[$1];
      else print $1, "---UNKNOWN---";
    }' > data/lang/$ds/words/word/te_words.txt;
    rm $tmp;
  }
  [ -s data/lang/$ds/words/word/te_queries.txt ] || {
    tmp=$(mktemp);
    awk '$1 == "<spot"' "data/xmls/${ds}_Test_GT_SegBased_QbE.xml" |
    sed -r 's|^.* word="([^"]+).jpg" image="([^"]+).jpg" .*$|\2 \1|g' |
    sort > "$tmp";
    join -1 1 data/lang/${ds}/words/word/te_words.txt "$tmp" |
    cut -d\  -f2-3 | awk '{print $2, $1}' |
    sort -u > data/lang/$ds/words/word/te_queries.txt;
    rm $tmp;
  }
done;

for d in 0 1; do
  ds=${DATASETS[d]};
  [ -s data/lang/${ds}/te_sb_qbe_rel_pairs.txt ] ||
  gawk '$1 == "<spot"{
    match($0, /^.* word="([a-z0-9]+).jpg" image="([a-z0-9]+).jpg".*$/, A);
    print A[1], A[2]
  }' data/xmls/${ds}_Test_GT_SegBased_QbE.xml \
       > data/lang/${ds}/te_sb_qbe_rel_pairs.txt;
done;
