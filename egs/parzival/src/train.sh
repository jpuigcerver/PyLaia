#!/bin/bash
export LC_NUMERIC=C;
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/..";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

source "$PWD/../utils/functions_check.inc.sh" || exit 1;
use_distortions=false;
max_epochs=200;
help_message="
Usage: ${0##*/} [options]

Options:
  --max_epochs       : (type = integer, default = $max_epochs)
  --use_distortions  : (type = boolean, default = $use_distortions)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

check_all_dirs data/parzivaldb-v1.0/data/line_images_normalized || exit 1;
check_all_files -s data/lang/char/syms_ctc.txt \
                   data/lang/char/tr.txt \
		   data/lang/char/va.txt || exit 1;

if [ "$use_distortions" = true ]; then
  ODIR="data/train_distortions";
else
  ODIR="data/train";
fi;

# Check whether or not training has been already completed.
ckpt="$ODIR/experiment.ckpt-$max_epochs";
if [ -s "$ckpt" ]; then
  msg="Checkpoint \"$ckpt\" exists. Do you want to overwrite it? (y/n) ";
  read -p "$msg" -n 1 -r; echo;
  if [[ $REPLY =~ ^[Cc]$ ]]; then
    rm -r "$ODIR";
  else
    echo "Aborted training..." >&2;
    exit 0;
  fi;
fi;

# Create model
mkdir -p "$ODIR";
../../pylaia-htr-create-model \
  --fixed_input_height=120 \
  --logging_also_to_stderr=info \
  --logging_file="$ODIR/train.log" \
  --logging_level=info \
  --logging_overwrite=true \
  --train_path=$ODIR \
  1 data/lang/char/syms_ctc.txt;

# Train model
../../pylaia-htr-train-ctc \
  --logging_also_to_stderr=info \
  --logging_file="$ODIR/train.log" \
  --logging_level=info \
  --logging_overwrite=false \
  --batch_size=16 \
  --learning_rate=0.0005 \
  --max_epochs="$max_epochs" \
  --train_path="$ODIR" \
  --show_progress_bar=true \
  --train_samples_per_epoch=6000 \
  --save_checkpoint_interval=5 \
  --use_distortions="$use_distortions" \
  data/lang/char/syms_ctc.txt \
  data/parzivaldb-v1.0/data/line_images_normalized \
  data/lang/char/tr.txt \
  data/lang/char/va.txt;
