#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/../..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../utils:$PATH";
export PATH="$PWD/../..:$PATH";
export LC_NUMERIC=C;

nbest=1000;
help_message="
Usage: ${0##*/} [options] syms_ctc lats_dir output_dir

Options:
  --nbest  : (type = integer, default = $nbest)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

syms_ctc="$1";
lats_dir="$2";
output_dir="$3";

check_all_files -s "$syms_ctc" \
		"$lats_dir/te.lat.ark" \
		"$lats_dir/va.lat.ark" \
		data/kws_line/lang/delimiters.txt || exit 1;

readarray -t delimiters < \
	  <(join -1 1 <(sort -k1b,1 data/kws_line/lang/delimiters.txt) \
	              <(sort -k1b,1 "$syms_ctc") |
            sort -nk2 | awk '{print $2 + 1}') || exit1;
echo "${delimiters[@]}"
mkdir -p "$output_dir" || exit 1;

exit 0;
for p in te va; do
  ./src/kws_line/compute_kws_metrics_char.py \
    --nbest="$nbest" \
    --acoustic-scale=1.0 \
    --queries=data/kws_line/lang/queries/iam_old_papers_queries.txt \
    --index-type=utterance \
    <(awk '{print $1, $2 + 1;}' "$syms_ctc") \
    data/kws_line/lang/kws_refs/$p.txt \
    "$lats_dir/$p.lat.ark" \
    "${delimiters[@]}" ||
    exit 1;

  continue;
  [ -s "$output_dir/$p.dat" ] ||
  lattice-char-index-segment \
    --num-threads="$(nproc)" --nbest="$nbest" \
    "${delimiters[*]}" "ark:$lats_dir/$p.lat.ark" "ark,t:$output_dir/$p.dat" ||
  exit 1;
done;
