#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../utils:$PATH";
export PATH="$PWD/../..:$PATH";

# Parse options
prior_scale=0.3;
help_message="
Usage: ${0##*/} [options] <train_dir> <images_dir> <output_dir>

Options:
  --prior_scale : (type = float, default = $prior_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;
train_dir="$1";
images_dir="$2";
output_dir="$3";

# Check for required files
check_all_files -s data/lang/syms_ctc.txt \
	               data/lang/lines/char/tr.txt \
	               data/lang/lines/char/va.txt || exit 1;
check_all_dirs "$train_dir" || exit 1;


############################################################
## 1. Compute pseudo-priors from the posteriors.
############################################################
mkdir -p "${output_dir}";
[ -s "${output_dir}/tr.prior" ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path "$train_dir" \
  --output_transform log_softmax \
  --output_format matrix \
  "$images_dir" \
  <(cut -d\  -f1 data/lang/lines/char/tr.txt) |
compute_ctc_priors.sh ark:- > "${output_dir}/tr.prior";


############################################################
## 2. Generate frame posteriors for validation text lines
############################################################
pst="$output_dir/va_ps0.0.mat";
[ -s "$pst.ark" -a -s "$pst.scp" ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path "$train_dir" \
  --output_transform log_softmax \
  --output_format matrix \
  "$images_dir" \
  <(cut -d\  -f1 "data/lang/lines/char/va.txt") |
  copy-matrix ark:- "ark,scp:$pst.ark,$pst.scp";


############################################################
## 3. Convert validation posteriors to pseudo-likelihoods
############################################################
mat="$output_dir/va_ps${prior_scale}.mat";
[ -s "$mat.ark" -a -s "$mat.scp" ] ||
convert_post_to_lkhs.sh \
  --scale "$prior_scale" "$output_dir/tr.prior" "ark:$pst.ark" ark:- |
add_boundary_frames.sh \
  "$(wc -l data/lang/syms_ctc.txt | awk '{print $1}')" \
  "$(grep "<sp>" data/lang/syms_ctc.txt | awk '{print $2}')" "" \
  ark:- "ark,scp:$mat.ark,$mat.scp";



exit 0;
############################################################
## 3. Generate frame posteriors for automatically segmented
##    text lines
############################################################
mat="data/bentham/lkhs/post/auto_segmented_lines.mat";
[ -s "$mat.ark" -a -s "$mat.scp" ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path data/bentham/train \
  --output_transform log_softmax \
  --output_format matrix \
  data/bentham/imgs/auto_segmented_lines_h80 \
  <(find data/bentham/imgs/auto_segmented_lines_h80 -name "*.png" | \
    xargs -n1 -I{} basename {} .png | sort -V) |
  copy-matrix ark:- "ark,scp:$mat.ark,$mat.scp";


############################################################
## 5. Convert frame posteriors to frame pseudo-likelihoods
##    for automatically segmented text lines
############################################################
pst="data/bentham/lkhs/post/auto_segmented_lines.mat.ark";
mat="data/bentham/lkhs/ps${prior_scale}/auto_segmented_lines.mat";
[ -s "$mat.ark" -a -s "$mat.scp" ] ||
convert_post_to_lkhs.sh \
  --scale "$prior_scale" data/bentham/lkhs/tr.prior "ark:$pst" ark:- |
add_boundary_frames.sh \
  "$(wc -l data/bentham/lang/syms_ctc.txt | awk '{print $1}')" \
  "$(grep "<sp>" data/bentham/lang/syms_ctc.txt | awk '{print $2}')" "" \
  ark:- "ark,scp:$mat.ark,$mat.scp";


############################################################
## 6. Convert frame posteriors to frame pseudo-likelihoods
##    for automatically segmented lines
############################################################
pst="data/bentham/lkhs/post/auto_segmented_lines.mat";
mat="data/bentham/lkhs/ps${prior_scale}/auto_segmented_lines.mat";
[ -s "$mat.ark" -a -s "$mat.scp" ] ||
convert_post_to_lkhs.sh \
  --scale "$prior_scale" data/bentham/lkhs/tr.prior "ark:$pst" ark:- |
add_boundary_frames.sh \
  "$(wc -l data/bentham/lang/syms_ctc.txt | awk '{print $1}')" \
  "$(grep "<sp>" data/bentham/lang/syms_ctc.txt | awk '{print $2}')" "" \
  ark:- "ark,scp:$mat.ark,$mat.scp";


############################################################
## 7. Generate frame likelihoods for image queries
############################################################
mat="data/bentham/lkhs/ps${prior_scale}/te.queries.mat";
[ -s "$mat.ark" -a -s "$mat.scp" ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path data/bentham/train \
  --output_transform log_softmax \
  --output_format matrix \
  data/bentham/imgs/queries_h80 \
  <(find data/bentham/imgs/queries_h80 -name "*.png" | xargs -n1 basename) |
convert_post_to_lkhs.sh \
  --scale "${prior_scale}" data/bentham/lkhs/tr.prior ark:- ark:- |
add_boundary_frames.sh \
  "$(wc -l data/bentham/lang/syms_ctc.txt | awk '{print $1}')" \
  "$(grep "<sp>" data/bentham/lang/syms_ctc.txt | awk '{print $2}')" "" \
  ark:- "ark,scp:$mat.ark,$mat.scp";
