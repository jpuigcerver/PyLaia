#!/bin/bash
set -e;
export PYTHONPATH=$PWD/../..:$PYTHONPATH;

train_path=$PWD/train_fixed_1;
rm -f "$train_path"/*;
mkdir -p "$train_path";

../../pylaia-htr-create-model \
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
  --train_path=$train_path \
  -- 78 1 syms1.txt;

../../pylaia-htr-train-ctc \
  --logging_also_to_stderr info \
  --logging_level info \
  --batch_size 1 \
  --learning_rate 0.001 \
  --gpu 0 \
  --train_path=$train_path \
  --train_samples_per_epoch=1 \
  -- syms1.txt imgs_h78 gt1.txt gt1.txt
