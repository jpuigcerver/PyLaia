#!/bin/bash
set -e;
export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../utils:$PATH";
export PATH="$PWD/../..:$PATH";

## TUNEABLE PARAMETERS
SCALE=0.3;

for f in data/bentham/lang/syms_ctc.txt \
         data/bentham/lang/char/te.txt \
	 data/bentham/lang/char/tr.txt \
	 data/bentham/lang/char/va.txt \
	 data/bentham/train/experiment.ckpt-80; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

############################################################
## 1. Compute pseudo-priors from the posteriors.
############################################################
mkdir -p data/bentham/post;
[ -s data/bentham/post/tr.lines_h80.prior ] ||
pylaia-htr-netout \
  --logging_also_to_stderr info \
  --logging_level info \
  --train_path data/bentham/train \
  --output_transform log_softmax \
  --output_format matrix \
  data/bentham/imgs/lines_h80 \
  <(cut -d\  -f1 data/bentham/lang/char/tr.txt) |
compute_ctc_priors.sh ark:- > data/bentham/post/tr.lines_h80.prior;


############################################################
## 2. Generate frame posteriors for text lines
############################################################
for p in te va; do
  mat=data/bentham/post/$p.lines_h80.mat;
  [ -s "$mat.ark" -a -s "$mat.scp" ] ||
  pylaia-htr-netout \
    --logging_also_to_stderr info \
    --logging_level info \
    --train_path data/bentham/train \
    --output_transform log_softmax \
    --output_format matrix \
    data/bentham/imgs/lines_h80 \
    <(cut -d\  -f1 data/bentham/lang/char/$p.txt) |
    copy-matrix ark:- "ark,scp:$mat.ark,$mat.scp";
done;


############################################################
## 3. Convert frame posteriors to frame pseudo-likelihoods
############################################################
mkdir -p data/bentham/lkhs;
for p in te va; do
  pst="data/bentham/post/$p.lines_h80.mat.ark";
  mat="data/bentham/lkhs/$p.lines_h80.s${SCALE}.mat";
  [ -s "$mat.ark" -a -s "$mat.scp" ] ||
  convert_post_to_lkhs.sh \
    --scale "$SCALE" data/bentham/post/tr.lines_h80.prior "ark:$pst" ark:- |
  add_boundary_frames.sh \
    "$(wc -l data/bentham/lang/syms_ctc.txt | awk '{print $1}')" \
    "$(grep "<sp>" data/bentham/lang/syms_ctc.txt | awk '{print $2}')" "" \
    ark:- "ark,scp:$mat.ark,$mat.scp";
done;


############################################################
## 4. Generate frame posteriors for image queries
############################################################
mat="data/bentham/lkhs/te.queries_h80.s${SCALE}.mat";
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
  --scale "$SCALE" data/bentham/post/tr.lines_h80.prior ark:- ark:- |
add_boundary_frames.sh \
  "$(wc -l data/bentham/lang/syms_ctc.txt | awk '{print $1}')" \
  "$(grep "<sp>" data/bentham/lang/syms_ctc.txt | awk '{print $2}')" "" \
  ark:- "ark,scp:$mat.ark,$mat.scp";
