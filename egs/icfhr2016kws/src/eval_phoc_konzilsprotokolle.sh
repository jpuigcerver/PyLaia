#!/bin/bash
set -e;
export LC_ALL=en_US.utf8;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -ne 2 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} MODEL_CHECKPOINT OUTPUT_FILE

Example: ${0##*/} train/konzilsprotokolle/phoc/r01/model.ckpt index/konzilsprotokolle/phoc/r01.dat
EOF
  exit 1;
fi;

QUERIES_TXT=data/lang/Konzilsprotokolle/words/word/te_queries.txt;
WORDS_TXT=data/lang/Konzilsprotokolle/words/word/te_words.txt;
RELEVANT_TXT=data/lang/Konzilsprotokolle/te_sb_qbe_rel_pairs.txt;

for f in "$QUERIES_TXT" "$WORDS_TXT" "$RELEVANT_TXT" "$1"; do
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

[ "$compute_dist" -eq 1 ] &&
python src/python/eval_phocnet.py \
       data/lang/Konzilsprotokolle/syms_ci_phoc.txt \
       data/images \
       <(awk '{$1="query/"$1; print;}' "$QUERIES_TXT") \
       <(awk '{$1="word/"$1; print;}' "$WORDS_TXT") \
       "$1" /dev/stdout |
       sed 's|query\/||g;s|word\/||g;' | sort -gk3 > "$2";

which SimpleKwsEval &> /dev/null &&
SimpleKwsEval \
  --collapse_matches true \
  --trapezoid_integral false \
  --interpolated_precision false \
  --sort none \
  "$RELEVANT_TXT" \
  "$2";
