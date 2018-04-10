#!/bin/bash
set -e;

for cv in cv1 cv2 cv3 cv4; do
  for p in tr te; do
    f="data/lang/dortmund/char/${cv}_${p}.txt";
    [ ! -s "$f" ] && echo "File \"$f\" not found!" >&2 && exit 1;
  done;
done;

lm=$(mktemp);
for n in 1 2 3 4 5 6 7 8 9 10; do
  for cv in cv1 cv2 cv3 cv4; do
    ngram-count -order $n -wbdiscount -interpolate \
		-text <(cut -d\  -f2- data/lang/dortmund/char/${cv}_tr.txt) \
		-lm $lm &> /dev/null;
    ngram -order $n -lm $lm \
	  -ppl <(cut -d\  -f2- data/lang/dortmund/char/${cv}_te.txt) |
      awk '$0 ~ /ppl= /' | sed -r 's|^.* ppl= ([0-9.]+) .*$|\1|g'
  done | awk -v n=$n '{ S += $1; }END{ print n, S / NR; }'
done;
