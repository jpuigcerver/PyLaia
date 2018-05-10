#!/usr/bin/env bash
set -e;
export LC_ALL=en_US.utf8;

if [ $# -lt 1 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} OUTPUT_DIR [TRAIN_OPTIONS]

Example: ${0##*/} train/dortmund/phocnet --gpu=2
EOF
  exit 1;
fi;

export PYTHONPATH=$PWD/../..:$PYTHONPATH;

TRAIN_TXT=data/kws/lang/words/char/tr.1.txt;
VALID_TXT=data/kws/lang/words/char/tr.2.txt;
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

# 5000 samples/epoch -> 500 updates/epoch
# 160 epochs needed for 80,000 updates.
python ./src/python/train_phocnet.py \
       --max_epochs=160 \
       --train_samples_per_epoch=5000 \
       --logging_also_to_stderr=INFO \
       --logging_file="$OUTPUT_DIR/train.log" \
       --train_path="$OUTPUT_DIR" \
       $@ \
       data/kws/lang/syms_phoc.txt \
       data/images/words \
       "$TRAIN_TXT" \
       "$VALID_TXT";
