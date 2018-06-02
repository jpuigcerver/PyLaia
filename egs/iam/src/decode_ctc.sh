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

for f in "$1" data/almazan/lang/syms_ctc.txt \
	      data/almazan/lang/char/va.txt \
	      data/almazan/lang/char/te.txt; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;
mkdir -p "$2";

for p in te va; do
  out="$2/${p}.txt";
  ask_owerwrite "$out" ||
  python src/python/decode_ctc.py \
	 --output_symbols=true \
	 data/almazan/lang/syms_ctc.txt \
	 data/original/words \
	 "data/almazan/lang/char/$p.txt" \
	 "$1" "$out";
  if which compute-wer &> /dev/null; then
    compute-wer "ark:data/almazan/lang/char/$p.txt" "ark:$out";
  fi;
done;
