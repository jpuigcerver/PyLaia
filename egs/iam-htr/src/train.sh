#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/src" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/src/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/src/parse_options.inc.sh file!" >&2 && exit 1;

# Model parameters
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
# Trainer parameters
batch_size=10;
learning_rate=0.0003;
gpu=1;
early_stop_epochs=20;
save_checkpoint_interval=10;
num_rolling_checkpoints=3;
show_progress_bar=true;
use_distortions=false;
img_directories="data/imgs/lines_h128";
checkpoint="ckpt.lowest-valid-cer*";
help_message="
Usage: ${0##*/} [options]

Options:
  --cnn_batchnorm            : (type = boolean list, default = \"$cnn_batchnorm\")
                               Batch normalization before the activation in each conv
                               layer.
  --cnn_dropout              : (type = double list, default = \"$cnn_dropout\")
                               Dropout probability at the input of each conv layer.
  --cnn_poolsize             : (type = integer list, default = \"$cnn_poolsize\")
                               Pooling size after each conv layer. It can be a list
                               of numbers if all the dimensions are equal or a list
                               of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_kernel_size          : (type = integer list, default = \"$cnn_kernel_size\")
                               Kernel size of each conv layer. It can be a list
                               of numbers if all the dimensions are equal or a list
                               of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_stride               : (type = integer list, default = \"$cnn_stride\")
                               Stride of each conv layer. It can be a list
                               of numbers if all the dimensions are equal or a list
                               of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_dilation             : (type = integer list, default = \"$cnn_dilation\")
                               Dilation of each conv layer. It can be a list
                               of numbers if all the dimensions are equal or a list
                               of strings formatted as tuples, e.g. (h1, w1) (h2, w2)
  --cnn_num_featuress        : (type = integer list, default = \"$cnn_num_features\")
                               Number of feature maps in each conv layer.
  --cnn_activations          : (type = string list, default = \"$cnn_activations\")
                               Type of the activation function in each conv layer,
                               valid types are \"ReLU\", \"Tanh\", \"LeakyReLU\".
  --rnn_layers               : (type = integer, default = $rnn_layers)
                               Number of recurrent layers.
  --rnn_units                : (type = integer, default = $rnn_units)
                               Number of units in the recurrent layers.
  --adaptive_poolingg        : (type = string, default = $adaptive_pooling)
                               Type of adaptive pooling to use, format:
                               {none,maxpool,avgpool}-[0-9]+
  --fixed_height             : (type = boolean, default = $fixed_height)
                               Use a fixed height model.
  --batch_size               : (type = integer, default = $batch_size)
                               Batch size for training.
  --learning_rate            : (type = float, default = $learning_rate)
                               Learning rate from RMSProp.
  --gpu                      : (type = integer, default = $gpu)
                               Select which GPU to use, index starts from 1.
                               Set to 0 for CPU.
  --early_stop_epochs        : (type = integer, default = $early_stop_epochs)
                               If n>0, stop training after this number of epochs
                               without a significant improvement in the validation CER.
                               If n=0, early stopping will not be used.
  --save_checkpoint_interval : (type=integer, default=$save_checkpoint_interval)
                               Make checkpoints of the training process every N epochs
  --num_rolling_checkpoints  : (type=integer, default=$num_rolling_checkpoints)
                               Keep this number of checkpoints during training
  --show_progress_bar        : (type=boolean, default=$show_progress_bar)
                               Whether or not to show a progress bar for each epoch
  --use_distortions          : (type=boolean, default=$use_distortions)
                               Whether or not to use distortions to augment
                               the training data
  --img_directories          : (type = string list, default = \"$img_directories\")
                               Image directories to use.
  --checkpoint               : (type = str, default = $checkpoint)
                               Suffix of the checkpoint to use, can be a glob pattern.
";
source "$(pwd)/src/parse_options.inc.sh" || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

if [ $gpu -gt 0 ]; then
  export CUDA_VISIBLE_DEVICES=$((gpu-1));
  gpu=1;
fi;

for f in data/lang/lines/char/aachen/{tr,va}.txt train/syms.txt; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

if [ $fixed_height = true ]; then
  extra_args="--fixed_input_height 128";
fi;

# Create model
pylaia-htr-create-model \
  1 "train/syms.txt" \
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
  $extra_args \
  --adaptive_pooling $adaptive_pooling \
  --logging_file "train/experiment.log" \
  --logging_also_to_stderr INFO \
  --train_path "train";

pylaia-htr-train-ctc \
  "train/syms.txt" \
  $img_directories \
  data/lang/lines/char/aachen/{tr,va}.txt \
  --gpu $gpu \
  --batch_size $batch_size \
  --max_nondecreasing_epochs $early_stop_epochs \
  --learning_rate $learning_rate \
  --save_checkpoint_interval $save_checkpoint_interval \
  --num_rolling_checkpoints $num_rolling_checkpoints \
  --show_progress_bar $show_progress_bar \
  --use_distortions $use_distortions \
  --checkpoint $checkpoint \
  --logging_file "train/experiment.log" \
  --logging_also_to_stderr INFO \
  --train_path "train";

exit 0;
