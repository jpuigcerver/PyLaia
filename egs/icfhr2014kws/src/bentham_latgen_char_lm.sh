#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

beam=25;
lattice_beam=15;
lazy_recipe=false;
ngram_order=7;
prior_scale=0.3;
help_message="
Usage: ${0##*/}
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

check_all_files -s "data/bentham/lkhs/ps${prior_scale}/te.lines_h80.mat.ark" \
                   "data/bentham/lkhs/ps${prior_scale}/va.lines_h80.mat.ark";

###########################################################
## 1. Build FSTs needed for generating the lattices
###########################################################
output_dir="data/bentham/lats/char_${ngram_order}gram";
[ "$lazy_recipe" = true ] && output_dir="${output_dir}_lz";
./src/bentham_build_char_lm_fsts.sh \
  --ngram_order "$ngram_order" \
  --lazy_recipe "$lazy_recipe" \
  "$output_dir" || exit 1;


###########################################################
## 2. Generate and align character lattices
###########################################################
mkdir -p "${output_dir}/ps${prior_scale}_b${beam}_lb${lattice_beam}";
for p in te va; do
  mat="data/bentham/lkhs/ps${prior_scale}/${p}.lines_h80.mat.ark";
  lat="${output_dir}/ps${prior_scale}_b${beam}_lb${lattice_beam}/${p}.lat";
  [[ -s "$lat.ark" && -s "$lat.scp" ]] || (
  latgen-faster-mapped-parallel \
    --acoustic-scale=1.0 \
    --beam="$beam" \
    --lattice-beam="$lattice_beam" \
    --num-threads=$(nproc) \
    --prune-interval=500 \
    "$output_dir/model" \
    "$output_dir/HCLG.fst" \
    "ark:$mat" ark:- |
  lattice-align-words-lexicon \
    "$output_dir/lexicon_align.txt" \
    "$output_dir/model" \
    ark:- "ark,scp:$lat.ark,$lat.scp";
  ) 2>&1 | tee "${output_dir}/ps${prior_scale}_b${beam}_lb${lattice_beam}/${p}.log" >&2;
done;
