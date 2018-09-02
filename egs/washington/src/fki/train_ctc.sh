#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";

cnn_num_features="16 32 64 64";
cnn_kernel_size="3 3 3 3";
cnn_stride="1 1 1 1";
cnn_dilation="1 1 1 1";
cnn_activations="LeakyReLU LeakyReLU LeakyReLU LeakyReLU";
cnn_poolsize="2 2 2 0";
cnn_dropout="0 0 0 0";
cnn_batchnorm="t t t t";
rnn_units=128;
rnn_layers=4;
adaptive_pooling="avgpool-16";
fixed_height=true;
gpu=1;
# Trainer parameters
batch_size=16;
max_epochs=700;
learning_rate=0.0003;
save_checkpoint_interval=10;
num_rolling_checkpoints=3;
show_progress_bar=true;
use_distortions=false;
help_message="
Usage: ${0##*/} [options] ctc_syms imgs_dir tr_txt va_txt output_dir
Options:
  --batch_size        : (type = integer, default = $batch_size)
                        Batch size for training.
  --cnn_batchnorm     : (type = boolean list, default = \"$cnn_batchnorm\")
                        Batch normalization before the activation in each conv
                        layer.
  --cnn_dropout       : (type = double list, default = \"$cnn_dropout\")
                        Dropout probability at the input of each conv layer.
  --cnn_poolsize      : (type = integer list, default = \"$cnn_poolsize\")
                        Pooling size after each conv layer. It can be a list
                        of numbers if all the dimensions are equal or a list
                        of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_kernel_size   : (type = integer list, default = \"$cnn_kernel_size\")
                        Kernel size of each conv layer. It can be a list
                        of numbers if all the dimensions are equal or a list
                        of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_stride        : (type = integer list, default = \"$cnn_stride\")
                        Stride of each conv layer. It can be a list
                        of numbers if all the dimensions are equal or a list
                        of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_dilation      : (type = integer list, default = \"$cnn_dilation\")
                        Dilation of each conv layer. It can be a list
                        of numbers if all the dimensions are equal or a list
                        of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_num_features  : (type = integer list, default = \"$cnn_num_features\")
                        Number of feature maps in each conv layer.
  --cnn_activations   : (type = string list, default = \"$cnn_activations\")
                        Type of the activation function in each conv layer,
                        valid types are \"ReLU\", \"Tanh\", \"LeakyReLU\".
  --rnn_layers        : (type = integer, default = $rnn_layers)
                        Number of recurrent layers.
  --rnn_units         : (type = integer, default = $rnn_units)
                        Number of units in the recurrent layers.
  --max_epochs        : (type = integer, default = $max_epochs)
                        Number of training epochs.
  --gpu               : (type = integer, default = $gpu)
                        Select which GPU to use, index starts from 1.
                        Set to 0 for CPU.
  --fixed_height      : (type = boolean, default = $fixed_height)
                        Use a fixed height model.
  --adaptive_pooling  : (type = string, default = $adaptive_pooling)
                        Type of adaptive pooling to use, format:
                        {none,maxpool,avgpool}-[0-9]+
  --learning_rate     : (type = float, default = $learning_rate)
                        Learning rate from RMSProp.
  --use_distortions   : (type = boolean, default = $use_distortions)
                        If true, train using data augmentation.
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -eq 5 ] || { echo "$help_message" >&2 && exit 1; }
mkdir -p "$5" || exit 1;

syms_ctc="$1";
imgs_dir="$2";
train_txt="$3";
valid_txt="$4";

if [ $gpu -gt 0 ]; then
  export CUDA_VISIBLE_DEVICES=$((gpu-1));
  gpu=1;
fi;

export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../..:$PATH";

extra_args=();
if [ "$fixed_height" = true ]; then
  extra_args+=("--fixed_input_height=120");
fi;

# Check required files
for f in "$syms_ctc" "$train_txt" "$valid_txt"; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;

[ -s "$5/experiment.ckpt-${max_epochs}" ] &&
echo "File \"$5/experiment.ckpt-${max_epochs}\" exists, cancel training!" >&2 && exit 0;

# Remove previous partial results
rm -f "$5/train.log" "$5/experiment.ckpt"*;

pylaia-htr-create-model \
  --cnn_num_features $cnn_num_features \
  --cnn_kernel_size $cnn_kernel_size \
  --cnn_stride $cnn_stride \
  --cnn_dilation $cnn_dilation \
  --cnn_activations $cnn_activations \
  --cnn_poolsize $cnn_poolsize \
  --cnn_dropout $cnn_dropout \
  --cnn_batchnorm $cnn_batchnorm \
  --rnn_units $rnn_units \
  --rnn_layers $rnn_layers \
  --adaptive_pooling $adaptive_pooling \
  --logging_file "$5/train.log" \
  --logging_also_to_stderr INFO \
  --train_path "$5" \
  "${extra_args[@]}" \
  -- \
  1 "$syms_ctc" || exit 1;

pylaia-htr-train-ctc \
  --gpu $gpu \
  --batch_size $batch_size \
  --max_epochs $max_epochs \
  --learning_rate $learning_rate \
  --save_checkpoint_interval $save_checkpoint_interval \
  --num_rolling_checkpoints $num_rolling_checkpoints \
  --show_progress_bar $show_progress_bar \
  --use_distortions $use_distortions \
  --logging_file "$5/train.log" \
  --logging_also_to_stderr INFO \
  --train_path "$5" \
  -- \
  "$syms_ctc" "$imgs_dir" "$train_txt" "$valid_txt" || exit 1;
