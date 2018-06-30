#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -ne 3 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PART CHECKPOINT OUTPUT_DIR

Example: ${0##*/} cv1 train/almazan/ctc/model.ckpt fsts/almazan/ctc
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

for f in "$2" \
	 data/lang/dortmund/syms_ctc.txt \
	 "data/lang/dortmund/word/${1}_te.txt" \
	 "data/lang/dortmund/${1}_rel_qbe.txt" \
	 ; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;
mkdir -p "$3";

oark="$3/${1}_te.mat.ark";
oscp="$3/${1}_te.mat.scp";
ask_owerwrite "$oark" "$oscp" ||
python src/python/generate_ctc_posteriors.py \
       --add_softmax \
       --add_boundary_blank \
       data/lang/dortmund/syms_ctc.txt \
       data/imgs/dortmund \
       <(join -1 1 \
	      <(sort "data/lang/dortmund/word/${1}_te.txt") \
	      <(cut -d\  -f1 "data/lang/dortmund/${1}_rel_qbe.txt"|sort -u)) \
       "$2" \
       >(copy-matrix ark:- "ark,scp:$oark,$oscp");
