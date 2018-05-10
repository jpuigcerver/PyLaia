#!/bin/bash
set -e;

if [ $# -lt 1 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} OUTPUT_DIR [TRAIN_OPTIONS]

Example: ${0##*/} train/almazan/phoc --gpu=2
EOF
  exit 1;
fi;

export PYTHONPATH=$PWD/../..:$PYTHONPATH;

TRAIN_TXT=data/almazan/lang/char/tr.txt;
VALID_TXT=data/almazan/lang/char/va_queries.txt;
OUTPUT_DIR="$1";
shift 1;

for f in "$TRAIN_TXT" "$VALID_TXT"; do
  [ -s "$f" ] || { echo "File \"$f\" wasn't found!" >&2 && exit 1; }
done;

mkdir -p "$OUTPUT_DIR";
ckpt="$OUTPUT_DIR/model.ckpt";
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

# 8000 samples/epoch -> 800 updates/epoch
# 300 epochs needed for 240,000 updates.
python ./src/python/train_phocnet.py \
       --max_epochs=300 \
       --train_samples_per_epoch=8000 \
       --logging_also_to_stderr=INFO \
       --logging_file="$OUTPUT_DIR/train.log" \
       --save_path="$OUTPUT_DIR" \
       $@ \
       data/almazan/lang/syms_phoc.txt \
       data/original/words \
       "$TRAIN_TXT" \
       <(sort -R --random-source="$VALID_TXT" "$VALID_TXT" | head -n3000);
