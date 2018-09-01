#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -lt 3 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PARTITION MODEL_CHECKPOINT OUTPUT_FILE [OPTIONS]

Example: ${0##*/} cv1 train/cv1/model.ckpt index/cv1.dat --gpu=2
EOF
  exit 1;
fi;

TXT=data/lang/dortmund/word/${1}_te.txt;
REL=data/lang/dortmund/${1}_rel_qbe.txt;
MODEL="$2";
for f in "$MODEL" "$TXT" "$REL"; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;
OUT="$3";
shift 3;

# Remove .gz extension from filename
if [ ${OUT:(-3)} = ".gz" ]; then
  OUT="${OUT%.*}";
fi;

mkdir -p "$(dirname "$OUT")";
compute_dist=1;
if [ -s "$OUT" -o -s "$OUT.gz" ]; then
  msg="Output \"$OUT\" already exists. Do you want to recompute it (y or n)? ";
  read -p "$msg" -n 1 -r; echo;
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    compute_dist=1;
  else
    compute_dist=0;
  fi;
fi;

[ "$compute_dist" -eq 1 ] && {
  python src/python/pairwise_phocnet.py \
	 --phoc_levels 1 2 3 4 5 \
	 --tpp_levels 1 2 3 4 5 \
	 --spp_levels \
	 "$@" \
	 -- \
	 data/lang/dortmund/syms_phoc.txt \
	 data/imgs/dortmund \
	 <(join -1 1 <(sort "$TXT") <(cut -d\  -f1 "$REL" | sort -u)) \
	 "$MODEL" "$OUT";
  rm -f "$OUT.gz";
  gzip -9 "$OUT";
}
[ ! -f "$OUT.gz" ] && { gzip -9 "$OUT"; }

which SimpleKwsEval &> /dev/null &&
( zcat "$OUT.gz" |
  SimpleKwsEval \
    --collapse_matches true \
    --trapezoid_integral false \
    --interpolated_precision false \
    --sort asc \
    "$REL" );
