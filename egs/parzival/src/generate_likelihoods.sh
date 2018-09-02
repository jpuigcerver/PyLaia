#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../utils:$PATH";
export PATH="$PWD/../..:$PATH";

# Parse options
prior_scale=0.3;
help_message="
Usage: ${0##*/} [options] train_dir output_dir
Options:
  --prior_scale : (type = float, default = $prior_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

train_dir="$1";
output_dir="$2";

# Check for all required files and directories
images_dir=data/parzivaldb-v1.0/data/line_images_normalized;
check_all_dirs "$images_dir" "$train_dir" || exit 1;
check_all_files -s data/lang/char/syms_ctc.txt \
                   data/lang/char/te.txt \
                   data/lang/char/tr.txt \
                   data/lang/char/va.txt;
mkdir -p "$output_dir" || exit 1;


############################################################
## 1. Compute pseudo-priors from the posteriors.
############################################################
[ -s "$output_dir/tr.prior" ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path "$train_dir" \
  --output_transform log_softmax \
  --output_format matrix \
  --show_progress_bar true \
  "$images_dir" \
  <(cut -d\  -f1 data/lang/char/tr.txt) |
  compute_ctc_priors.sh ark:- > "$output_dir/tr.prior";


############################################################
## 2. Generate frame posteriors for text lines
############################################################
mkdir -p "$output_dir/post";
for p in te va; do
  mat="$output_dir/post/$p.mat";
  [ -s "$mat.ark" -a -s "$mat.scp" ] ||
  pylaia-htr-netout \
    --logging_also_to_stderr info \
    --logging_level info \
    --train_path "$train_dir" \
    --output_transform log_softmax \
    --output_format matrix \
    --show_progress_bar true \
    "$images_dir" \
    <(cut -d\  -f1 "data/lang/char/$p.txt") |
    copy-matrix ark:- "ark,scp:$mat.ark,$mat.scp";
done;


############################################################
## 3. Convert frame posteriors to frame pseudo-likelihoods
############################################################
mkdir -p "$output_dir/ps${prior_scale}";
for p in te va; do
  pst="$output_dir/post/$p.mat.ark";
  mat="$output_dir/ps${prior_scale}/$p.mat";
  [ -s "$mat.ark" -a -s "$mat.scp" ] ||
  convert_post_to_lkhs.sh \
    --scale "$prior_scale" "$output_dir/tr.prior" "ark:$pst" ark:- |
  add_boundary_frames.sh \
    "$(wc -l data/lang/char/syms_ctc.txt | awk '{print $1}')" \
    "$(grep "<space>" data/lang/char/syms_ctc.txt | awk '{print $2}')" "" \
    ark:- "ark,scp:$mat.ark,$mat.scp";
done;
