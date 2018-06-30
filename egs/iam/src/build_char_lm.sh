#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
source "$SDIR/functions.inc.sh";

ngram_opts=(-wbdiscount);

mkdir -p decode/almazan/char_lm;

arpa=decode/almazan/char_lm/lm.arpa;
! ask_owerwrite "$arpa" || exit 0;

best_ppl=100000; best_n=0;
tmpd="$(mktemp -d)";
for n in 3 4 5 6 7 8; do
  cut -d\  -f2- data/almazan/lang/char/tr.txt |
  ngram-count -order $n "${ngram_opts[@]}" -text - -lm "$tmpd/$n.arpa";
  ppl=$(cut -d\  -f2- data/almazan/lang/char/va.txt |
	ngram -order $n -ppl - -lm "$tmpd/$n.arpa" |
	grep "ppl" | sed -r 's|^.*ppl1= ([0-9.]+)$|\1|g');
  if [ "$(echo "$ppl < $best_ppl" | bc -l)" -eq 1 ]; then
    best_ppl="$ppl";
    best_n="$n";
  fi;
done;
echo "Best order on validation: $best_n (ppl1 = $best_ppl)" >&2;

cat data/almazan/lang/char/{tr,va}.txt | cut -d\  -f2- |
ngram-count -order "$best_n" "${ngram_opts[@]}" -text - -lm "$arpa";

ppl=$(cut -d\  -f2- data/almazan/lang/char/te.txt |
      ngram -order "$best_n" -ppl - -lm "$arpa" |
      grep "ppl" | sed -r 's|^.*ppl1= ([0-9.]+)$|\1|g');
echo "Test ppl1 = $ppl" >&2;
rm -r "$tmpd";
