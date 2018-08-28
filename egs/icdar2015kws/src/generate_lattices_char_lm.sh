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
ngram_order=7;
prior_scale=0.3;
query_nbest=100;
help_message="
Usage: ${0##*/} [Options] <lkhs_dir> <graph_dir> <lats_dir>

Options:
  --beam         : (type = float, default = $beam)
  --lattice_beam : (type = float, default = $lattice_beam)
  --ngram_order  : (type = integer, default = $ngram_order)
  --prior_scale  : (type = float, default = $prior_scale)
  --query_nbest  : (type = integer, default = $query_nbest)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

lkhs_dir="$1";
graph_dir="$2";
lats_dir="$3";
check_all_files -s "$lkhs_dir/va_ps${prior_scale}.mat.ark";


###########################################################
## 1. Build FSTs needed for generating the line lattices
###########################################################
./src/prepare_char_lm_fsts.sh \
  --ngram_order "$ngram_order" "$graph_dir" || exit 1;


###########################################################
## 2. Generate and align character validation lattices
###########################################################
mkdir -p "$lats_dir";
lkh="$lkhs_dir/va_ps${prior_scale}.mat.ark";
lat="${lats_dir}/va_ps${prior_scale}.lat";
[[ -s "$lat.ark" && -s "$lat.scp" ]] || (
latgen-faster-mapped-parallel \
  --acoustic-scale=1.0 \
  --beam="$beam" \
  --lattice-beam="$lattice_beam" \
  --num-threads=$(nproc) \
  --prune-interval=500 \
  "$graph_dir/model" \
  "$graph_dir/HCLG.fst" \
  "ark:$lkh" ark:- |
lattice-align-words-lexicon \
  "$graph_dir/lexicon_align.txt" \
  "$graph_dir/model" \
  ark:- "ark,scp:$lat.ark,$lat.scp";
) 2>&1 | tee "${lats_dir}/va.log" >&2;


###########################################################
## 3. Generate and align character lattices for auto
##    segmented text lines
###########################################################
for p in te va; do
  lkh="${lkhs_dir}/${p}_autosegm_ps${prior_scale}.mat.ark";
  lat="${lats_dir}/${p}_autosegm_ps${prior_scale}.lat";
  [[ -s "$lat.ark" && -s "$lat.scp" ]] || (
    latgen-faster-mapped-parallel \
      --acoustic-scale=1.0 \
      --beam="$beam" \
      --lattice-beam="$lattice_beam" \
      --num-threads=$(nproc) \
      --prune-interval=500 \
      "$graph_dir/model" \
      "$graph_dir/HCLG.fst" \
      "ark:$lkh" ark:- |
    lattice-align-words-lexicon \
      "$graph_dir/lexicon_align.txt" \
      "$graph_dir/model" \
      ark:- "ark,scp:$lat.ark,$lat.scp";
  ) 2>&1 | tee "${lats_dir}/${p}_autosegm.log" >&2;
done;
