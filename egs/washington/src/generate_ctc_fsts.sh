#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -ne 3 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PARTITION_ID CHECKPOINT OUTPUT_DIR

Example: ${0##*/} cv1 train/dortmund/cv1/model.ckpt /data
EOF
  exit 1;
fi;

[ -d "$3" ] || mkdir -p "$3";

MAXBEAM=15;
TXT=data/lang/dortmund/word/${1}_te.txt;
REL=data/lang/dortmund/${1}_rel_qbe.txt;
for f in "$2" "$TXT" "$REL"; do
  [ ! -s "$f" ] && echo "File \"$f\" was not found!" >&2 && exit 1;
done;

[ -s "$3/lat.ark" ] ||
python src/python/generate_ctc_lattice.py \
       --add_softmax \
       data/lang/dortmund/syms_ctc.txt \
       data/imgs/dortmund \
       <(join -1 1 <(sort "$TXT") <(cut -d\  -f1 "$REL" | sort -u)) \
         "$2" \
	 >(lattice-remove-ctc-blank 1 ark:- ark:- | \
	   lattice-prune --beam=$MAXBEAM ark:- "ark:$3/lat.ark") || exit 1;

# Get 1-best path
[ -s "$3/b0.fst.ark" -a -s "$3/b0.fst.scp" ] ||
lattice-1best "ark:$3/lat.ark" ark:- |
lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
	       ark:- "ark,scp:$3/b0.fst.ark,$3/b0.fst.scp" || exit 1;


# Prune for different beam thresholds
#for b in $(seq $MAXBEAM); do
for b in $MAXBEAM; do
  [ -s "$3/b$b.fst.ark" -a -s "$3/b$b.fst.scp" ] ||
  lattice-prune --beam=$b "ark:$3/lat.ark" ark:- |
  lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
		 ark:- "ark,scp:$3/b$b.fst.ark,$3/b$b.fst.scp" || exit 1;
done;
