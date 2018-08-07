#!/usr/bin/env bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export LC_NUMERIC=C;
export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../..:$PATH";

batch_size=16;
cnn_activations="ReLU ReLU ReLU ReLU";
cnn_dilation="1 1 1 1";
cnn_kernel_size="7 5 3 3";
cnn_num_features="12 24 48 48";
cnn_pool_size="2 2 2 0";
cnn_stride="1 1 1 1";
height=80;
learning_rate=0.0005;
max_epochs=90;
rnn_units=256;
rnn_layers=3;
help_message="
Usage: ${0##*/} [options] <output_dir>

Options:
  --batch_size       : (type = integer, default = $batch_size)
  --cnn_activation   : (type = string, default = \"$cnn_activations\")
  --cnn_dilation     : (type = integer, default = \"$cnn_dilation\")
  --cnn_kernel_size  : (type = integer, default = \"$cnn_kernel_size\")
  --cnn_num_features : (type = integer, default = \"$cnn_num_features\")
  --cnn_pool_size    : (type = integer, default = \"$cnn_pool_size\")
  --cnn_stride       : (type = integer, default = \"$cnn_stride\")
  --height           : (type = integer, default = $height)
  --learning_rate    : (type = float, default = $learning_rate)
  --max_epochs       : (type = integer, default = $max_epochs)
  --rnn_layers       : (type = integer, default = $rnn_layers)
  --rnn_units        : (type = integer, default = $rnn_units)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

# Check required files
check_all_dirs data/images/train_lines_proc_h${height} || exit 1;
check_all_files -s data/lang/syms_ctc.txt \
                   data/lang/lines/char/tr.txt \
                   data/lang/lines/char/va.txt || exit 1;

output_dir="$1";
# Ask for permission to overwrite previous run
confirm_overwrite_all_files "${output_dir}/experiment.ckpt-${max_epochs}" ||
{ echo "INFO: Abort training..." >&2 && exit 0; }

rm -rf "${output_dir}"; # Remove old runs
mkdir -p "${output_dir}";

pylaia-htr-create-model \
  --logging_also_to_stderr info \
  --logging_file "${output_dir}/train.log" \
  --logging_level info \
  --logging_overwrite true \
  --train_path "${output_dir}" \
  --fixed_input_height "$height" \
  --cnn_activations ${cnn_activations} \
  --cnn_dilation ${cnn_dilation} \
  --cnn_kernel_size ${cnn_kernel_size} \
  --cnn_num_features ${cnn_num_features} \
  --cnn_poolsize ${cnn_pool_size} \
  --cnn_stride ${cnn_stride} \
  --rnn_layers "$rnn_layers" \
  --rnn_units "$rnn_units" \
  --rnn_dropout 0.5 \
  --lin_dropout 0.5 \
  --use_masked_conv false \
  -- \
  1 data/lang/syms_ctc.txt;

pylaia-htr-train-ctc \
  --logging_also_to_stderr info \
  --logging_file "${output_dir}/train.log" \
  --logging_level info \
  --logging_overwrite false \
  --train_path "${output_dir}" \
  --learning_rate "${learning_rate}" \
  --train_samples_per_epoch 5000 \
  --delimiters "<sp>" \
  --max_epochs "${max_epochs}" \
  --save_checkpoint_interval 5 \
  --batch_size ${batch_size} \
  --show_progress_bar true \
  -- \
  data/lang/syms_ctc.txt \
  "data/images/train_lines_proc_h${height}" \
  data/lang/lines/char/{tr,va}.txt;
