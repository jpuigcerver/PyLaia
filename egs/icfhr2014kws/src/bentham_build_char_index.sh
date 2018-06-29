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

latdir="data/bentham/lats/char_${ngram_order}gram/ps${prior_scale}_b${beam}_lb${lattice_beam}";
chars_file="data/bentham/lats/char_${ngram_order}gram/chars.txt";
check_all_files "$latdir/te.lat.ark" \
                "$latdir/va.lat.ark" \
                "$chars_file" || exit 1;

wspace="$(grep "<sp>" "$chars_file" | awk '{print $2}' | tr \\n \ )";
marks="$(egrep "^[!?] " "$chars_file" | awk '{print $2}' | tr \\n \ )";
paren="$(egrep "^(\(|\)|\[|\]) " "$chars_file" | awk '{print $2}' | tr \\n \ )";

for p in te; do
  [ -s "$latdir/$p.pos.index" ] ||
  lattice-char-index-position \
    --nbest=10000 \
    --num-threads=$(nproc) \
    --acoustic-scale=1.1 \
    --other-groups="$marks ; $paren" "$wspace" \
    "ark:$latdir/$p.lat.ark" \
    "ark,t:$latdir/$p.pos.index";

  [ -s "$latdir/$p.seg.index" ] ||
  lattice-char-index-segment \
    --nbest=10000 \
    --num-threads=$(nproc) \
    --acoustic-scale=1.1 \
    --other-groups="$marks ; $paren" "$wspace" \
    "ark:$latdir/$p.lat.ark" \
    "ark,t:$latdir/$p.seg.index";

  kws_index_frame2box.sh \
    -s ${scale} \
    -x ${offset} \
    -D <(matrix-dim --print-args=false \
         ark:data/bentham/lkhs/ps${prior_scale}/${p}.lines_h80.mat.ark | \
         awk '{ print $1, $2, 80; }') \
    position \
    "$latdir/$p.pos.index";
done;

