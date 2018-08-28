#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

offset=-10;
query_nbest=100;
scale=8;
acoustic_scale=1.0;
height=128;
manual_scale_width=1.2;
manual_scale_height=2.0;
prior_scale=0.5;
help_message="
Usage: ${0##*/} [options] <fst_dir> <lats_dir> <indx_dir>

Options:
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;
fst_dir="$1";
lats_dir="$2";
indx_dir="$3";

subs_dir="${indx_dir}/submissions";
resize_info=("data/images/autosegm/te_h$height/resize_info.txt" \
	     "data/images/autosegm/va_h$height/resize_info.txt" );
fpgrams_info=(data/images/autosegm/te/fpgrams.txt \
              data/images/autosegm/va/fpgrams.txt);

check_all_files -s data/lang/delimiters.txt \
		   "$fst_dir/chars.txt" \
                   "$lats_dir/te_autosegm_ps${prior_scale}.lat.ark" \
		   "$lats_dir/va_autosegm_ps${prior_scale}.lat.ark" \
		   "${resize_info[@]}" \
		   "${fpgrams_info[@]}" || exit 1;

mkdir -p "$indx_dir" "$subs_dir";

readarray -t delims < <(join -1 1 \
			     <(sort -k1b,1 "$fst_dir/chars.txt") \
			     <(sort -k1b,1 data/lang/delimiters.txt) | \
                        sort -nk2 | awk '{print $2}');

part=(te va);
qbs_queries=(data/Test_Query_Keywords.lst data/Validation_QueryByString.lst);
for i in $(seq 1 ${#part[@]}); do
  p="${part[i-1]}";
  [ -s "$indx_dir/${p}_autosegm.index" ] ||
  lattice-char-index-segment \
    --nbest=50 \
    --num-threads="$(nproc)" \
    --acoustic-scale="${acoustic_scale}" \
    "${delims[*]}" \
    "ark:$lats_dir/${p}_autosegm_ps${prior_scale}.lat.ark" \
    "ark,t:$indx_dir/${p}_autosegm.index" || exit 1;

  [ -s "$subs_dir/${p}_qbs.txt" ] ||
  extract_kws_index_bounding_box.py \
    --output_bounding_box \
    --resize_info_file="${resize_info[i-1]}" \
    --symbols_table="$fst_dir/chars.txt" \
    --global_scale=8 \
    --global_shift=-10 \
    segment \
    "${indx_dir}/${p}_autosegm.index" \
    "${fpgrams_info[i-1]}" \
    "data/images/autosegm/${p}_h${height}" |
    awk '{ print $1, toupper($6), $2, $3, $4, $5, $7; }' |
    sort -k2b,2 |
    join -1 2 - <(sort -k1b,1 "${qbs_queries[i-1]}") |
    awk '{ print $2, $1, $3, $4, $5, $6, $7; }' > "${subs_dir}/${p}_qbs.txt";
done;
