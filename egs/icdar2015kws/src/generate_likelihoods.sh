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
Usage: ${0##*/} [options] <train_dir> <train_imgs> <te_autosegm_imgs> <va_autosegm_imgs> <output_dir>

Options:
  --prior_scale : (type = float, default = $prior_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 5 ] && echo "$help_message" >&2 && exit 1;
train_dir="$1";
images_dir1="$2";
images_dir2="$3";
images_dir3="$4";
output_dir="$5";

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
  --show_progress_bar true \
  "$images_dir1" \
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
  --show_progress_bar true \
  "$images_dir1" \
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


############################################################
## 4. Generate frame posteriors for automatically segmented
##    text lines
############################################################
mat=("$output_dir/te_autosegm_ps0.0.mat" "$output_dir/va_autosegm_ps0.0.mat");
imgdir=("$images_dir2" "$images_dir3");
for i in $(seq 1 ${#mat[@]}); do
  [ -s "${mat[i-1]}.ark" -a -s "${mat[i-1]}.scp" ] ||
  pylaia-htr-netout \
    --logging_also_to_stderr info \
    --logging_level info \
    --train_path "$train_dir" \
    --output_transform log_softmax \
    --output_format matrix \
    --show_progress_bar true \
    "${imgdir[i-1]}" \
    <(find "${imgdir[i-1]}" -name "*.png" |
      xargs -n1 -I{} basename {} .png | \
      sort -V) |
  copy-matrix ark:- "ark,scp:${mat[i-1]}.ark,${mat[i-1]}.scp";
done;


############################################################
## 5. Convert frame posteriors to frame pseudo-likelihoods
##    for automatically segmented text lines
############################################################
pst=("$output_dir/te_autosegm_ps0.0.mat.ark" \
     "$output_dir/va_autosegm_ps0.0.mat.ark");
mat=("$output_dir/te_autosegm_ps${prior_scale}.mat" \
     "$output_dir/va_autosegm_ps${prior_scale}.mat");
for i in $(seq 1 ${#pst[@]}); do
  [ -s "${mat[i-1]}.ark" -a -s "${mat[i-1]}.scp" ] ||
  convert_post_to_lkhs.sh \
    --scale "$prior_scale" "${output_dir}/tr.prior" "ark:${pst[i-1]}" ark:- |
  add_boundary_frames.sh \
    "$(wc -l data/lang/syms_ctc.txt | awk '{print $1}')" \
    "$(grep "<sp>" data/lang/syms_ctc.txt | awk '{print $2}')" "" \
    ark:- "ark,scp:${mat[i-1]}.ark,${mat[i-1]}.scp" || exit 1;
done;
