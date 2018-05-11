#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -ne 2 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} CHECKPOINT OUTPUT_DIR

Example: ${0##*/} train/almazan/ctc/model.ckpt fsts/almazan/ctc
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

maxbeam=20;
[ ! -s "$1" ] && echo "File \"$1\" does not exist!" >&2 && exit 1;
mkdir -p "$2";

FILES=(
  data/almazan/lang/word/te_queries.txt
  data/almazan/lang/word/te_distractors.txt
);
PREFIX=(q d);
SUBSET=(te te);
for i in 0 1; do
  s="${SUBSET[i]}"
  p="${PREFIX[i]}";
  ask_owerwrite "$2/${s}_${p}.lat.ark" ||
  python src/python/generate_ctc_lattice.py \
	 --add_softmax \
	 data/almazan/lang/syms_ctc.txt \
	 data/original/words \
	 "${FILES[i]}" "$1" \
	 >(lattice-remove-ctc-blank 1 ark:- ark:- |
	   lattice-prune "--beam=$maxbeam" ark:- "ark:$2/${s}_${p}.lat.ark");

  ask_owerwrite "$2/${s}_${p}_b0.fst.ark" "$2/${s}_${p}_b0.fst.scp" ||
  lattice-1best "ark:$2/${s}_${p}.lat.ark" ark:- |
  lattice-to-fst \
    --acoustic-scale=1 --lm-scale=1 \
    ark:- "ark,scp:$2/${s}_${p}_b0.fst.ark,$2/${s}_${p}_b0.fst.scp";

  # Prune for different beam thresholds
  for b in $(seq $maxbeam); do
    ask_owerwrite "$2/${s}_${p}_b${b}.fst.ark" "$2/${s}_${p}_b${b}.fst.scp" ||
    lattice-prune --beam=$b "ark:$2/${s}_${p}.lat.ark" ark:- |
    lattice-to-fst \
      --acoustic-scale=1 --lm-scale=1 \
      ark:- "ark,scp:$2/${s}_${p}_b${b}.fst.ark,$2/${s}_${p}_b${b}.fst.scp";
  done;
done;
