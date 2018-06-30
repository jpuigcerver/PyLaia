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

# Check for required files
check_all_files -s data/bentham/lang/syms_ctc.txt \
                   data/bentham/lang/char/te.txt \
	               data/bentham/lang/char/tr.txt \
	               data/bentham/lang/char/va.txt \
	               data/bentham/train/experiment.ckpt-90;

# Parse options
prior_scale=0.3;
help_message="
Usage: ${0##*/} [options]

Options:
  --prior_scale : (type = float, default = $prior_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;


############################################################
## 1. Compute pseudo-priors from the posteriors.
############################################################
mkdir -p data/bentham/lkhs;
[ -s data/bentham/lkhs/tr.prior ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path data/bentham/train \
  --output_transform log_softmax \
  --output_format matrix \
  data/bentham/imgs/lines_h80 \
  <(cut -d\  -f1 data/bentham/lang/char/tr.txt) |
compute_ctc_priors.sh ark:- > data/bentham/lkhs/tr.prior;


############################################################
## 2. Generate frame posteriors for text lines
############################################################
mkdir -p data/bentham/lkhs/post;
for p in te va; do
  mat="data/bentham/lkhs/post/$p.mat";
  [ -s "$mat.ark" -a -s "$mat.scp" ] ||
  pylaia-htr-netout \
    --logging_also_to_stderr info \
    --logging_level info \
    --train_path data/bentham/train \
    --output_transform log_softmax \
    --output_format matrix \
    data/bentham/imgs/lines_h80 \
    <(cut -d\  -f1 "data/bentham/lang/char/$p.txt") |
    copy-matrix ark:- "ark,scp:$mat.ark,$mat.scp";
done;


############################################################
## 3. Convert frame posteriors to frame pseudo-likelihoods
############################################################
mkdir -p data/bentham/lkhs/ps${prior_scale};
for p in te va; do
  pst="data/bentham/lkhs/post/$p.mat.ark";
  mat="data/bentham/lkhs/ps${prior_scale}/$p.mat";
  [ -s "$mat.ark" -a -s "$mat.scp" ] ||
  convert_post_to_lkhs.sh \
    --scale "$prior_scale" data/bentham/lkhs/tr.prior "ark:$pst" ark:- |
  add_boundary_frames.sh \
    "$(wc -l data/bentham/lang/syms_ctc.txt | awk '{print $1}')" \
    "$(grep "<sp>" data/bentham/lang/syms_ctc.txt | awk '{print $2}')" "" \
    ark:- "ark,scp:$mat.ark,$mat.scp";
done;


############################################################
## 4. Generate frame likelihoods for image queries
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
