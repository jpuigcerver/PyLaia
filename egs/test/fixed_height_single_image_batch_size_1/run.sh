#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR";
export PYTHONPATH="$PWD/../../..:$PYTHONPATH";

train_path="$PWD";
rm -f train.log *.ckpt* trainer model;

../../../pylaia-htr-create-model \
  --logging_also_to_stderr info \
  --logging_file train.log \
  --logging_level info \
  --cnn_num_features 16 32 \
  --cnn_kernel_size 3 3 \
  --cnn_stride 1 1 \
  --cnn_dilation 1 1 \
  --cnn_poolsize 2 2 \
  --cnn_dropout 0 0 \
  --cnn_activations ReLU ReLU \
  --cnn_batchnorm false false \
  --rnn_units 64 \
  --rnn_dropout 0 \
  --lin_dropout 0 \
  --fixed_input_height 78 \
  --train_path="$train_path" \
  -- 1 syms.txt;

../../../pylaia-htr-train-ctc \
  --logging_also_to_stderr info \
  --logging_file train.log \
  --logging_level info \
  --batch_size 1 \
  --learning_rate 0.001 \
  --gpu 0 \
  --max_epochs 500 \
  --train_path="$train_path" \
  -- syms.txt ../imgs_h78 gt.txt gt.txt
