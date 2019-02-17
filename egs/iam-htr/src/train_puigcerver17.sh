#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Move to the top directory of the experiment.
cd "$(dirname "${BASH_SOURCE[0]}")/..";

source ../utils/functions_check.inc.sh || exit 1;

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
fixed_height=128;
# Trainer parameters
add_logsoftmax_to_loss=false;
batch_size=10;
checkpoint="ckpt.lowest-valid-cer*";
early_stop_epochs=20;
gpu=1;
img_directories="data/imgs/lines_h128";
learning_rate=0.0003;
num_rolling_checkpoints=3;
save_checkpoint_interval=10;
show_progress_bar=true;
use_baidu_ctc=true;
use_distortions=false;
help_message="
Usage: ${0##*/} [options]

Options:
  --add_logsoftmax_to_loss   : (type = boolean, default = $add_logsoftmax_to_loss)
                               If true, add a logsoftmax operation to the CTC loss.
  --adaptive_pooling         : (type = string, default = $adaptive_pooling)
                               Type of adaptive pooling to use, format:
                               {none,maxpool,avgpool}-[0-9]+
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
  --fixed_height             : (type = integer, default = $fixed_height)
                               Use a fixed height model.
  --batch_size               : (type = integer, default = $batch_size)
                               Batch size for training.
  --learning_rate            : (type = float, default = $learning_rate)
                               Learning rate from RMSProp.
  --use_baidu_ctc            : (type = boolean, default = $use_baidu_ctc)
                               If true, use Baidu's CTC implementation.
  --gpu                      : (type = integer, default = $gpu)
                               Select which GPU to use, index starts from 1.
                               Set to 0 for CPU.
  --early_stop_epochs        : (type = integer, default = $early_stop_epochs)
                               If n>0, stop training after this number of epochs
                               without a significant improvement in the validation CER.
                               If n=0, early stopping will not be used.
  --save_checkpoint_interval : (type=integer, default=$save_checkpoint_interval)
                               Make checkpoints of the training process every N epochs.
  --num_rolling_checkpoints  : (type=integer, default=$num_rolling_checkpoints)
                               Keep this number of checkpoints during training.
  --show_progress_bar        : (type=boolean, default=$show_progress_bar)
                               Whether or not to show a progress bar for each epoch.
  --use_distortions          : (type=boolean, default=$use_distortions)
                               Whether or not to use distortions to augment the training data.
  --img_directories          : (type = string list, default = \"$img_directories\")
                               Image directories to use. If more than one, separate them with
                               spaces.
  --checkpoint               : (type = str, default = $checkpoint)
                               Suffix of the checkpoint to use, can be a glob pattern.
";
source "../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

if [ $gpu -gt 0 ]; then
  export CUDA_VISIBLE_DEVICES=$((gpu-1));
  gpu=1;
fi;

check_all_files \
  data/lang/puigcerver/lines/char/tr.txt \
  data/lang/puigcerver/lines/char/va.txt;

mkdir -p exper/puigcerver17/train;
[ -s exper/puigcerver17/train/syms_ctc.txt ] ||
cut -d\  -f2- data/lang/puigcerver/lines/char/{tr,va}.txt | tr \  \\n |
sort -u | awk 'BEGIN{ print "<ctc>", 0; }{ print $1, NR; }' \
  > exper/puigcerver17/train/syms_ctc.txt

extra_args=();
if [ -n "$fixed_height" ]; then
  extra_args+=(--fixed_input_height "$fixed_height");
fi;

# Create model
pylaia-htr-create-model \
  1 "exper/puigcerver17/train/syms_ctc.txt" \
  --adaptive_pooling "$adaptive_pooling" \
  --cnn_num_features $cnn_num_features \
  --cnn_kernel_size $cnn_kernel_size \
  --cnn_stride $cnn_stride \
  --cnn_dilation $cnn_dilation \
  --cnn_activations $cnn_activations \
  --cnn_poolsize $cnn_poolsize \
  --cnn_dropout $cnn_dropout \
  --cnn_batchnorm $cnn_batchnorm \
  --rnn_units "$rnn_units" \
  --rnn_layers "$rnn_layers" \
  --logging_file "exper/puigcerver17/train/log" \
  --logging_also_to_stderr INFO \
  --train_path "exper/puigcerver17/train" \
  "${extra_args[@]}";

# Train
pylaia-htr-train-ctc \
  "exper/puigcerver17/train/syms_ctc.txt" \
  $img_directories \
  data/lang/puigcerver/lines/char/{tr,va}.txt \
  --add_logsoftmax_to_loss "$add_logsoftmax_to_loss" \
  --batch_size "$batch_size" \
  --checkpoint "$checkpoint" \
  --delimiters "@" \
  --gpu "$gpu" \
  --learning_rate "$learning_rate" \
  --logging_also_to_stderr INFO \
  --logging_file "exper/puigcerver17/train/log" \
  --max_nondecreasing_epochs "$early_stop_epochs" \
  --num_rolling_checkpoints "$num_rolling_checkpoints" \
  --save_checkpoint_interval "$save_checkpoint_interval" \
  --show_progress_bar "$show_progress_bar" \
  --train_path "exper/puigcerver17/train" \
  --use_baidu_ctc "$use_baidu_ctc" \
  --use_distortions "$use_distortions";


exit 0;
