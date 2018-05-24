#!/bin/bash
set -e;
export LC_NUMERIC=C;

CHECK_FILES=(
  data/almazan/lang/word/te_queries.txt
  data/almazan/lang/word/te_distractors.txt
  decode/almazan/char_lm/lats/te.ark
);
for f in "${CHECK_FILES[@]}"; do
  [ ! -f "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

for a in $(seq 1.5 0.1 3.0); do
  oscp=fsts/almazan/ctc/r01/epoch-300/te_a${a}.scp;
  oark="${oscp/.scp/.ark}";
  lattice-to-fst --acoustic-scale=$a --lm-scale=1 \
		 ark:decode/almazan/char_lm/lats/te.ark \
		 "ark,scp:$oark,$oscp";
  join -1 1 \
       <(cut -d\  -f1 data/almazan/lang/word/te_distractors.txt | sort) \
       <(sort "$oscp") > fsts/almazan/ctc/r01/epoch-300/te_d_a${a}.scp;
  join -1 1 \
       <(cut -d\  -f1 data/almazan/lang/word/te_queries.txt | sort) \
       <(sort "$oscp") > fsts/almazan/ctc/r01/epoch-300/te_q_a${a}.scp;
done;
