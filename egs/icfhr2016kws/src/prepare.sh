#!/bin/bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

mkdir -p data/images/word;
DATASETS=(Botany Konzilsprotokolle);
WPREFIX=(bt kt);
for d in 0 1; do
  ds=${DATASETS[d]};
  mkdir -p data/lang/${ds}/words/word;
  [ -s data/lang/${ds}/words/word/tr.txt ] || {
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
    awk '{ print $1, $2 }' "$tmp" > "data/lang/${ds}/words/word/tr.txt";
  }
  [ -s data/lang/${ds}/words/word/tr_ci.txt ] || {
    tmp=$(mktemp);
    for s in I II III; do
      awk '$1 == "<spot"' data/xmls/${ds}_Train_${s}_WL_CASE_INSENSITIVE.xml
    done |
    sed -r 's|^.* word="([^"]+)" .*$|\1|g' |
    awk -v wp=${WPREFIX[d]} '{
      printf("%s%06d %s\n", wp, NR, toupper($1));
    }' > data/lang/${ds}/words/word/tr_ci.txt;
  }
done;
