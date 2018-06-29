#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";

beam=25;
lattice_beam=15;
ngram_order=7;
prior_scale=0.3;
scale=8;
offset=-10;
help_message="
Usage: ${0##*/} [options]

Options:
  --ngram_order  : (type = int, default = $ngram_order)
  --prior_scale  : (type = float, default = $prior_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

base_dir="data/bentham/decode/char_${ngram_order}gram";
lats_dir="$base_dir/lats/ps${prior_scale}_b${beam}_lb${lattice_beam}";
indx_dir="$base_dir/indx/ps${prior_scale}_b${beam}_lb${lattice_beam}";
check_all_files "$lats_dir/te.lat.ark" \
                "$lats_dir/va.lat.ark" \
                "$base_dir/chars.txt" || exit 1;

wspace="$(grep "<sp>" "$base_dir/chars.txt" | awk '{print $2}' | tr \\n \ )";
marks="$(egrep "^[!?] " "$base_dir/chars.txt" | awk '{print $2}' | tr \\n \ )";
paren="$(egrep "^(\(|\)|\[|\]) " "$base_dir/chars.txt" | awk '{print $2}' | tr \\n \ )";

mkdir -p "$indx_dir";
for p in te va; do
  [ -s "$indx_dir/$p.pos.index" ] ||
  lattice-char-index-position \
    --nbest=10000 \
    --num-threads=$(nproc) \
    --acoustic-scale=1.1 \
    --other-groups="$marks ; $paren" "$wspace" \
    "ark:$lats_dir/$p.lat.ark" \
    "ark,t:$indx_dir/$p.pos.index";

  [ -s "$indx_dir/$p.seg.index" ] ||
  lattice-char-index-segment \
    --nbest=10000 \
    --num-threads=$(nproc) \
    --acoustic-scale=1.1 \
    --other-groups="$marks ; $paren" "$wspace" \
    "ark:$lats_dir/$p.lat.ark" \
    "ark,t:$indx_dir/$p.seg.index";

  #kws_index_frame2box.sh \
  #  -s ${scale} \
  #  -x ${offset} \
  #  -D <(matrix-dim --print-args=false \
  #       ark:data/bentham/lkhs/ps${prior_scale}/${p}.lines_h80.mat.ark | \
  #       awk '{ print $1, $2, 80; }') \
  #  position \
  #  "$lats_dir/$p.pos.index";
done;
