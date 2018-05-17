#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -lt 2 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PARTITION_ID OUTPUT_DIR [TRAIN_OPTIONS]

Example: ${0##*/} cv1 train/dortmund/ctc/cv1 --gpu=2
EOF
  exit 1;
fi;

TRAIN_TXT="data/lang/dortmund/char/${1}_tr.txt";
VALID_TXT="data/lang/dortmund/char/${1}_te.txt";
OUTPUT_DIR="$2";
shift 2;

for f in "$TRAIN_TXT" "$VALID_TXT"; do
  [ ! -s "$f" ] && echo "File \"$f\" was not found!" >&2 && exit 1;
done;

mkdir -p "$OUTPUT_DIR";
ckpt="$OUTPUT_DIR/model.ckpt-160";
if [ -s "$ckpt" ]; then
    msg="Checkpoint \"$ckpt\" already exists. Continue (c) or abort (a)? ";
    read -p "$msg" -n 1 -r; echo;
    if [[ $REPLY =~ ^[Cc]$ ]]; then
       :
    else
        echo "Aborted training..." >&2;
        exit 0;
    fi;
fi;

python ./steps/train_ctc.py \
       --max_epochs=160 \
       --train_samples_per_epoch=5000 \
       --logging_also_to_stderr=INFO \
       --logging_file="$OUTPUT_DIR/train.log" \
       --train_path="$OUTPUT_DIR" \
       $@ \
       data/lang/dortmund/syms_ctc.txt \
       data/imgs/dortmund \
       "$TRAIN_TXT" \
       "$VALID_TXT";
