#!/bin/bash
export LC_NUMERIC=C;
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/..";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

# Check whether or not training has been already completed.
ckpt=data/train/experiment.ckpt-100;
if [ -s "$ckpt" ]; then
  msg="Checkpoint \"$ckpt\" exists. Do you want to overwrite (y/n) it? ";
  read -p "$msg" -n 1 -r; echo;
  if [[ $REPLY =~ ^[Cc]$ ]]; then
    rm -r data/train;
  else
    echo "Aborted training..." >&2;
    exit 0;
  fi;
fi;

# Create model
mkdir -p data/train;
../../pylaia-htr-create-model \
  --fixed_input_height=120 \
  --logging_also_to_stderr=info \
  --logging_file=data/train/train.log \
  --logging_level=info \
  --logging_overwrite=true \
  --train_path=data/train \
  1 data/lang/char/syms.txt;

# Train model
../../pylaia-htr-train-ctc \
  --logging_also_to_stderr=info \
  --logging_file=data/train/train.log \
  --logging_level=info \
  --logging_overwrite=false \
  --batch_size=16 \
  --learning_rate=0.0005 \
  --max_epochs=100 \
  --train_path=data/train \
  --show_progress_bar=true \
  --train_samples_per_epoch=6000 \
  --save_checkpoint_interval=5 \
  data/lang/char/syms.txt \
  data/parzivaldb-v1.0/data/line_images_normalized \
  data/lang/char/tr.txt \
  data/lang/char/va.txt;
