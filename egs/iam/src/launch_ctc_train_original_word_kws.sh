#!/bin/bash
set -e;

if [ $# -lt 1 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} OUTPUT_DIR [TRAIN_OPTIONS]

Example: ${0##*/} train/original/word_kws/ctc --gpu=2
EOF
  exit 1;
fi;

export PYTHONPATH=$PWD/../..:$PYTHONPATH;

TRAIN_TXT=data/original/word_kws/lang/char/tr.txt;
VALID_TXT=data/original/word_kws/lang/char/va.txt;
OUTPUT_DIR="$1";
shift 1;

for f in "$TRAIN_TXT" "$VALID_TXT"; do
  [ -s "$f" ] || { echo "File \"$f\" wasn't found!" >&2 && exit 1; }
done;

mkdir -p "$OUTPUT_DIR";

if [ -s "$OUTPUT_DIR/model.ckpt" ]; then
    ckpt="$OUTPUT_DIR/model.ckpt";
    msg="Checkpoint \"$ckpt\" already exists. Continue (c) or abort (a)? ";
    read -p "$msg" -n 1 -r; echo;
    if [[ $REPLY =~ ^[Cc]$ ]]; then
       :
    else
        echo "Aborted training..." >&2;
        exit 0;
    fi;
fi;

# 3645 is the number of train samples in Washington.
# 658 is the number of epochs to perform ~240000 updates.
# This allows to get stats as frequently as in the Washington experiment
# while making and comparable with the IAM setting of the PHOCNet paper.
python ./src/python/train_ctc.py \
       --max_epochs=660 \
       --train_samples_per_epoch=3645 \
       --valid_samples_per_epoch=6000 \
       --logging_also_to_stderr=INFO \
       --logging_file="$OUTPUT_DIR/train.log" \
       --save_path="$OUTPUT_DIR" \
       $@ \
       data/original/word_kws/lang/syms_ctc.txt \
       data/original/words \
       "$TRAIN_TXT" \
       "$VALID_TXT";
