#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -ne 2 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} CHECKPOINT OUTPUT_DIR

Example: ${0##*/} train/almazan/phoc/r01/model.ckpt index/almazan/phoc
EOF
  exit 1;
fi;

function ask_owerwrite () {
  local exist=0;
  local expect="$#";
  local f="$1";
  while [ $# -gt 0 ]; do
    if [ -s "$1" ]; then ((++exist)); fi;
    shift 1;
  done;
  if [ "$exist" -eq "$expect" ]; then
    msg="File \"$f\" already exists. Do you want to overwrite it (y or n) ? ";
    read -p "$msg" -n 1 -r; echo;
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      return 1;
    else
      return 0;
    fi;
  fi;
  return 1;
}

[ ! -s "$1" ] && echo "File \"$1\" does not exist!" >&2 && exit 1;
mkdir -p "$2";

FILES=(
  data/almazan/lang/word/te_queries.txt
  data/almazan/lang/word/te_distractors.txt
);
PREFIX=(q d);
for i in 0 1; do
  p="${PREFIX[i]}";
  ask_owerwrite "$2/lat.$p.ark" ||
  python src/python/generate_phoc_lattice.py \
	 --add_sigmoid \
	 data/almazan/lang/syms_phoc.txt \
	 data/original/words \
	 "${FILES[i]}" "$1" >(lattice-copy ark:- "ark:$2/lat.$p.ark");
  # Full fst, no pruning
  ask_owerwrite "$2/fst.$p.ark" "$2/fst.$p.scp" ||
  lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
                 "ark:$2/lat.$p.ark" "ark,scp:$2/fst.$p.ark,$2/fst.$p.scp";
  # Beam pruning = 0 (equiv to 1-best)
  ask_owerwrite "$2/fst_b0.$p.ark" "$2/fst_b0.$p.scp" ||
  lattice-1best "ark:$2/lat.$p.ark" ark:- |
  lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
                 ark:- "ark,scp:$2/fst_b0.$p.ark,$2/fst_b0.$p.scp";
  for b in $(seq 20); do
    ask_owerwrite "$2/fst_b$b.$p.ark" "$2/fst_b$b.$p.scp" ||
    lattice-prune --beam=$b "ark:$2/lat.$p.ark" ark:- |
    lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
                   ark:- "ark,scp:$2/fst_b$b.$p.ark,$2/fst_b$b.$p.scp";
  done;
done;
