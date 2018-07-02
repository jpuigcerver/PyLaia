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
query_nbest=100;
help_message="
Usage: ${0##*/}
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

check_all_files -s "data/bentham/lkhs/ps${prior_scale}/te.mat.ark" \
                   "data/bentham/lkhs/ps${prior_scale}/va.mat.ark";


###########################################################
## 1. Build FSTs needed for generating the lattices
###########################################################
output_dir="data/bentham/decode/char_${ngram_order}gram";
: <<EOF
[ "$lazy_recipe" = true ] && output_dir="${output_dir}_lz";
./src/bentham_build_char_lm_fsts.sh \
  --ngram_order "$ngram_order" \
  --lazy_recipe "$lazy_recipe" \
  "$output_dir" || exit 1;
EOF

###########################################################
## 2. Generate and align character lattices
###########################################################
lats_dir="${output_dir}/lats/ps${prior_scale}_b${beam}_lb${lattice_beam}";
mkdir -p "$lats_dir";
for p in te va; do
  mat="data/bentham/lkhs/ps${prior_scale}/${p}.mat.ark";
  lat="${lats_dir}/${p}.lat";
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
  ) 2>&1 | tee "${lats_dir}/${p}.log" >&2;
done;


###########################################################
## 3. Generate and align character lattices for auto
##    segmented text lines
###########################################################
mat="data/bentham/lkhs/ps${prior_scale}/auto_segmented_lines.mat.ark";
lat="${lats_dir}/auto_segmented_lines.lat";
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
) 2>&1 | tee "${lats_dir}/auto_segmented_lines.log" >&2;


###########################################################
## 4. Build FSTs needed for generating the query lattices
###########################################################
output_dir="data/bentham/decode/query_char_${ngram_order}gram";
[ "$lazy_recipe" = true ] && output_dir="${output_dir}_lz";
./src/bentham_build_query_fsts.sh \
  --ngram_order "$ngram_order" \
  "$output_dir" || exit 1;


###########################################################
## 5. Generate and align character lattices for queries
###########################################################
lats_dir="${output_dir}/lats/ps${prior_scale}_b${beam}_lb${lattice_beam}";
mkdir -p "$lats_dir";
mat="data/bentham/lkhs/ps${prior_scale}/te.queries.mat.ark";
lat="${lats_dir}/queries.lat";
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
) 2>&1 | tee "${lats_dir}/queries.log" >&2;


###########################################################
## 6. Generate n-best list for queries
###########################################################
[ -s "${output_dir}/queries.${query_nbest}best.txt" ] || {
  tmp="$(mktemp)";
  lattice-to-nbest \
    --n="$query_nbest" --acoustic-scale=0.8 \
    "ark:$lat.ark"  ark:- |
  lattice-best-path \
    --acoustic-scale=0.8 \
    ark:- \
    "ark,t:| ../utils/int2sym.pl -f 2- ${output_dir}/chars.txt" \
    2> >(grep "For utterance" | sed -r 's|^.*For utterance ([^,]+), .* = ([0-9.]+) .*$|\1 \2|g' | sort -V > "$tmp.2") |
  awk '{$2=""; s=""; for(i=2;i<=NF;++i) { s=s""$i; } print $1, s; }' |
  sort -V > "$tmp.1";

  join -1 1 "$tmp.1" "$tmp.2" |
  awk '
  function logadd(a, b) {
    if (b > a) { t = a; a = b; b = t; }
    return a + log(1.0 + exp(b - a));
  }
  function print_entry(query, nbest, costs, total) {
    n1 = split(nbest, nbest_arr, " ");
    n2 = split(costs, costs_arr, " ");
    if (n1 != n2) { print "This should not happen!" > "/dev/stderr"; exit(1); }
    printf("%s", query);
    for (i = 1; i <= n1; ++i) {
      printf(" %s %g", nbest_arr[i], costs_arr[i] - total);
    }
    printf("\n");
  }

  BEGIN{
    prev_id = "";
    NBEST = "";
    COSTS = "";
    TOTAL = 0;
  }{
    split($1, a, "-");
    query_id = a[1];
    word = $2;
    cost = $3;

    if (prev_id != "" && query_id != prev_id) {
      print_entry(prev_id, NBEST, COSTS, TOTAL);
    }

    if (query_id == prev_id) {
      NBEST = NBEST" "word;
      COSTS = COSTS" "(-cost);
      TOTAL = logadd(TOTAL, -cost);
    } else {
      NBEST = word;
      COSTS = -cost;
      TOTAL = -cost;
    }

    prev_id = query_id;
  }END{
    if (prev_id != "") {
      print_entry(prev_id, NBEST, COSTS, TOTAL);
    }
  }' |
  sed -r 's|(sf[0-9]+)\.png|\1|g' \
    > "${output_dir}/queries.${query_nbest}best.txt" || exit 1;
  rm -f "$tmp"*;
}
