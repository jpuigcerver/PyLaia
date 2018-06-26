#!/bin/bash
set -e;
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

for f in data/bentham/lang/syms_ctc.txt \
	 data/bentham/lang/char/tr.txt \
	 data/bentham/lang/char/va.txt; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

if [ -f train/bentham/model -o -f train/bentham/trainer ]; then
  msg="The directory contains training data. Do you want to overwrite it?";
  read -p "$msg" -n 1 -r; echo;
  if [[ "$REPLY" =~ "^[Yy]$" ]]; then
    rm -rf train/bentham; # Remove old runs
  else
    echo "Abort training..." >&2;
    exit 0;
  fi;
fi;
mkdir -p train/bentham;

../../pylaia-htr-create-model \
  --logging_also_to_stderr info \
  --logging_file train/bentham/train.log \
  --logging_level info \
  --logging_overwrite true \
  --train_path train/bentham \
  --fixed_input_height 80 \
  --cnn_num_features 12 24 48 48 \
  --cnn_kernel_size 7 5 3 3 \
  --cnn_stride 1 1 1 1 \
  --cnn_dilation 1 1 1 1 \
  --cnn_activations ReLU ReLU ReLU ReLU \
  --cnn_poolsize 2 2 2 0 \
  --rnn_units 256 \
  --rnn_layers 3 \
  --rnn_dropout 0.5 \
  --lin_dropout 0.5 \
  --use_masked_conv false \
  -- \
  1 data/bentham/lang/syms_ctc.txt;

../../pylaia-htr-train-ctc \
  --logging_also_to_stderr info \
  --logging_file train/bentham/train.log \
  --logging_level info \
  --logging_overwrite false \
  --train_path train/bentham \
  --learning_rate 0.0005 \
  --train_samples_per_epoch 5000 \
  --delimiters "<sp>" \
  --max_epochs 80 \
  --save_checkpoint_interval 5 \
  --batch_size 16 \
  --show_progress_bar true \
  -- \
  data/bentham/lang/syms_ctc.txt \
  data/bentham/imgs/lines_h80 \
  data/bentham/lang/char/{tr,va}.txt;
