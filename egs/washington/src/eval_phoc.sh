#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -ne 3 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PARTITION MODEL_CHECKPOINT OUTPUT_FILE

Example: ${0##*/} cv1 train/cv1/model.ckpt index/cv1.dat
EOF
  exit 1;
fi;

TXT=data/lang/dortmund/word/${1}_te.txt;
REL=data/lang/dortmund/${1}_rel_qbe.txt;
for f in "$2" "$TXT" "$REL"; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;

mkdir -p "$(dirname "$3")";
compute_dist=1;
if [ -s "$3" ]; then
  msg="Output \"$3\" already exists. Do you want to recompute it (y or n)? ";
  read -p "$msg" -n 1 -r; echo;
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    compute_dist=1;
  else
    compute_dist=0;
  fi;
fi;

[ "$compute_dist" -eq 1 ] &&
python src/python/pairwise_phocnet.py \
       --phoc_levels 1 2 3 4 5 \
       --tpp_levels 1 2 3 4 5 \
       --spp_levels \
       -- \
       data/lang/dortmund/syms_phoc.txt \
       data/imgs/dortmund \
       <(join -1 1 <(sort "$TXT") <(cut -d\  -f1 "$REL" | sort -u)) \
       "$2" "$3";

which SimpleKwsEval &> /dev/null &&
SimpleKwsEval \
  --collapse_matches true \
  --trapezoid_integral false \
  --interpolated_precision false \
  --sort asc \
  "$REL" \
  "$3";
