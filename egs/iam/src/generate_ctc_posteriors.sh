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

[ ! -s "$1" ] && echo "File \"$1\" does not exist!" >&2 && exit 1;
mkdir -p "$2";

for p in te va; do
  oark="$2/${p}.mat.ark";
  oscp="$2/${p}.mat.scp";
  ask_owerwrite "$oark" "$oscp" ||
  python src/python/generate_ctc_posteriors.py \
	 --add_softmax \
	 --add_boundary_blank \
	 data/almazan/lang/syms_ctc.txt \
	 data/original/words \
	 "data/almazan/lang/word/$p.txt" \
	 "$1" \
	 >(copy-matrix ark:- "ark,scp:$oark,$oscp");
done;
