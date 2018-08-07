#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";

cnn_num_features="16 32 48 64 80";
cnn_kernel_size="3 3 3 3 3";
cnn_stride="1 1 1 1 1";
cnn_dilation="1 1 1 1 1";
cnn_activations="LeakyReLU LeakyReLU LeakyReLU LeakyReLU LeakyReLU";
cnn_poolsize="2 2 2 0 0";
cnn_dropout="0 0 0 0 0";
cnn_batchnorm="f f f f f";
rnn_units=256;
rnn_layers=5;
adaptive_pooling="avgpool-16";
fixed_height=true;
gpu=1;
# Trainer parameters
batch_size=16;
max_epochs=120;
learning_rate=0.0003;
save_checkpoint_interval=10;
num_rolling_checkpoints=3;
show_progress_bar=true;
lowercase=false;
help_message="
Usage: ${0##*/} [options] imgs_dir output_dir
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
  --lowercase         : (type = boolean, default = $lowercase)
                        If true, train lowercase-only model.
";
source ../utils/parse_options.inc.sh || exit 1;
[ $# -eq 2 ] || { echo "$help_message" >&2 && exit 1; }
mkdir -p "$2" || exit 1;

if [ $gpu -gt 0 ]; then
  export CUDA_VISIBLE_DEVICES=$((gpu-1));
  gpu=1;
fi;

export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../..:$PATH";

extra_args=();
if [ $fixed_height = true ]; then
  extra_args+=("--fixed_input_height=128");
fi;

if [ "$lowercase" = true ]; then
  syms_ctc=data/kws_line/lang/char/syms_ctc_lowercase.txt;
  train_txt=data/kws_line/lang/char/tr_lowercase.txt;
  valid_txt=data/kws_line/lang/char/va_lowercase.txt;
else
  syms_ctc=data/kws_line/lang/char/syms_ctc.txt;
  train_txt=data/kws_line/lang/char/tr.txt;
  valid_txt=data/kws_line/lang/char/va.txt;
fi;

# Check required files
for f in "$syms_ctc" "$train_txt" "$valid_txt"; do
  [ ! -s "$f" ] && echo "File \"$f\" does not exist!" >&2 && exit 1;
done;

[ -s "$2/experiment.ckpt-${max_epochs}" ] &&
echo "File \"$2/experiment.ckpt-${max_epochs}\" exists, cancel training!" >&2 && exit 0;

# Remove previous partial results
rm -f "$2/train.log" "$2/experiment.ckpt"*;

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
  --logging_file "$2/train.log" \
  --logging_also_to_stderr INFO \
  --train_path "$2" \
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
  --logging_file "$2/train.log" \
  --logging_also_to_stderr INFO \
  --train_path "$2" \
  -- \
  "$syms_ctc" "$1" "$train_txt" "$valid_txt" || exit 1;
