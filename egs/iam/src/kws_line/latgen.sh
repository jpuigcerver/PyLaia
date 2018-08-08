#!/usr/bin/env bash
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/../..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";

beam=25;
determinize_lattice=true;
lattice_beam=15;
phone_determinize=true;
word_determinize=true;
help_message="
${0##*/} [options] lkhs_dir fsts_dir output_dir

Options:
  --determinize_lattice : (type = boolean, default = $determinize_lattice)
                          If true, determinize the lattice (lattice-determinization,
                          keeping only best pdf-sequence for each word-sequence).
  --beam                : (type = float, default = $beam)
                          Decoding beam. Larger->slower, more accurate.
  --lattice_beam        : (type = float, default = $lattice_beam)
                          Lattice generation beam. Larger->slower, and deeper lattices.
  --phone_determinize   : (type = boolean, default = $phone_determinize)
                          If true, do an initial pass of determinization on both phones
                          and words (see also --word_determinize).
  --word_determinize    : (type = boolean, default = $word_determinize)
                          If true, do a second pass of determinization on words only
                          (see also --phone_determinize).
";
source ../utils/parse_options.inc.sh || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

lkhs_dir="$1";
fsts_dir="$2";
output_dir="$3";

check_kaldi || exit 1;
check_all_dirs "$lkhs_dir" "$fsts_dir" || exit 1;
check_all_files -s "$lkhs_dir/va.mat.ark" "$lkhs_dir/va.mat.scp" \
                   "$lkhs_dir/te.mat.ark" "$lkhs_dir/te.mat.scp" \
                   "$fsts_dir/lexicon_align.txt" \
                   "$fsts_dir/model" "$fsts_dir/HCLG.fst" || exit 1;
mkdir -p "$output_dir" || exit 1;


for p in te va; do
  mat="$lkhs_dir/${p}.mat.ark";
  lat="${output_dir}/${p}.lat";
  [[ -s "$lat.ark" && -s "$lat.scp" ]] || (
  latgen-faster-mapped-parallel \
    --acoustic-scale=1.0 \
    --beam="$beam" \
    --determinize-lattice="$determinize_lattice" \
    --lattice-beam="$lattice_beam" \
    --num-threads=$(nproc) \
    --phone-determinize="$phone_determinize" \
    --prune-interval=500 \
    --word-determinize="$word_determinize" \
    "$fsts_dir/model" \
    "$fsts_dir/HCLG.fst" \
    "ark:$mat" ark:- |
  lattice-align-words-lexicon \
    "$fsts_dir/lexicon_align.txt" \
    "$fsts_dir/model" \
    ark:- "ark,scp:$lat.ark,$lat.scp";
  ) 2>&1 | tee "${output_dir}/${p}.log" >&2;
done;
