#!/bin/bash
set -e;

if [ $# -ne 3 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PARTITION_ID CHECKPOINT OUTPUT_DIR

Example: ${0##*/} cv1 train/dortmund/cv1/model.ckpt /data
EOF
  exit 1;
fi;

export PYTHONPATH=$HOME/src/PyLaia:$PYTHONPATH;

cv="$1";
model="$2";
outdir="$3";

[ -d "$outdir" ] || mkdir -p "$outdir";

[ -s "$outdir/$cv.lat.ark" -a -s "$outdir/$cv.lat.scp" ] || {
  python steps/generate_ctc_lattice.py --add_softmax \
	 train/dortmund/syms.txt \
	 data/imgs/dortmund \
	 "data/lang/dortmund/char/${cv}_te.txt" \
	 "$model" \
	 >(lattice-remove-ctc-blank \
	     1 ark:- "ark,scp:$outdir/$cv.lat.ark,$outdir/$cv.lat.scp");
}

# Get 1-best path
fn="$outdir/${cv}_b0.fst";
[ -s "$fn.ark" -a -s "$fn.scp" ] || {
  join -1 1 <(sort queries_$cv.lst) <(sort "$outdir/$cv.lat.scp") |
  lattice-1best scp:- ark:- |
  lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
		 ark:- "ark,scp:$fn.ark,$fn.scp";
}

# Prune for different beam thresholds
for beam in $(seq 20); do
  fn="$outdir/${cv}_b${beam}.fst";
  [ -s "$fn.ark" -a -s "$fn.scp" ] || {
    join -1 1 <(sort queries_$cv.lst) <(sort "$outdir/$cv.lat.scp") |
    lattice-prune --beam=$beam scp:- ark:- |
    lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
		   ark:- "ark,scp:$fn.ark,$fn.scp";
  }
done;
