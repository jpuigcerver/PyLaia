#!/bin/bash
set -e;
export LC_ALL=en_US.utf8;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -lt 2 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} MODEL_CHECKPOINT OUTPUT_FILE [...]

Example: ${0##*/} train/almazan/phoc/r01/model.ckpt index/almazan/phoc/r01.dat
EOF
  exit 1;
fi;

QUERIES_TXT=data/almazan/lang/word/te_queries.txt;
DISTRACTORS_TXT=data/almazan/lang/word/te_distractors.txt;
RELEVANT_TXT=data/almazan/te_qbe_rel_pairs.txt;
for f in "$1" "$QUERIES_TXT" "$DISTRACTORS_TXT" "$RELEVANT_TXT"; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;

mkdir -p "$(dirname "$2")";

compute_dist=1;
if [ -s "$2" ]; then
  msg="Output \"$2\" already exists. Do you want to recompute it (y or n)? ";
  read -p "$msg" -n 1 -r; echo;
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    compute_dist=1;
  else
    compute_dist=0;
  fi;
fi;

model="$1";
output="$2";
shift 2;

[ "$compute_dist" -eq 1 ] &&
python src/python/pairwise_prob_phoc.py \
       --distractors="$DISTRACTORS_TXT" \
       $@ \
       data/almazan/lang/syms_phoc.txt \
       data/original/words \
       "$QUERIES_TXT" "$model" /dev/stdout |
if [ "${output:(-3)}" = ".gz" ]; then
  gzip -9 -c;
elif [ "${output:(-4)}" = ".bz2" ]; then
  bzip -9 -c;
else
  cat;
fi > "$output";

which SimpleKwsEval &> /dev/null &&
if [ "${output:(-3)}" = ".gz" ]; then
  zcat "$output";
elif [ "${output:(-4)}" = ".bz2" ]; then
  bzcat "$output";
else
  cat "$output";
fi | SimpleKwsEval \
  --collapse_matches true \
  --trapezoid_integral false \
  --interpolated_precision false \
  --sort desc \
  "$RELEVANT_TXT";
