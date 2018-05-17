#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

if [ $# -lt 1 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} OUTPUT_DIR [TRAIN_OPTIONS]

Example: ${0##*/} train/almazan/ctc --gpu=2
EOF
  exit 1;
fi;


SYMS=data/almazan/lang/syms_ctc.txt;
TRAIN_TXT=data/almazan/lang/char/tr.txt;
VALID_TXT=data/almazan/lang/char/va_queries.txt;
OUTPUT_DIR="$1";
shift 1;

for f in "$SYMS" "$TRAIN_TXT" "$VALID_TXT"; do
  [ -s "$f" ] || { echo "File \"$f\" wasn't found!" >&2 && exit 1; }
done;

mkdir -p "$OUTPUT_DIR";


../../pylaia-htr-create-model \
    --cnn_num_features 16 32 48 64 \
    --cnn_poolsize 2 2 0 0 \
    --rnn_units=256 \
    --rnn_layers=4 \
    --train_path="$OUTPUT_DIR" \
    -- 64 1 "$SYMS";

# 8000 samples/epoch -> 800 updates/epoch
# 300 epochs needed for 240,000 updates.
./src/python/train_ctc2.py \
    --batch_size=10 \
    --max_epochs=300 \
    --train_samples_per_epoch=8000 \
    --logging_also_to_stderr=INFO \
    --logging_file="$OUTPUT_DIR/train.log" \
    --train_path="$OUTPUT_DIR" \
    $@ \
    64 "$SYMS" data/original/words \
    "$TRAIN_TXT" \
    <(sort -R --random-source="$VALID_TXT" "$VALID_TXT" | head -n3000);
