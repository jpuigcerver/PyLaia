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

lattice_beam=15;
help_message="
Usage: ${0##*/} [options] syms_ctc images_dir train_dir output_dir
Options:
  --lattice_beam : (type = float, default = $lattice_beam)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 4 ] && echo "$help_message" >&2 && exit 1;

syms_ctc="$1";
images_dir="$2";
train_dir="$3";
output_dir="$4";

check_all_files -s "$syms_ctc" \
	           data/kws_line/lang/char/te.txt \
                   data/kws_line/lang/char/va.txt || exit 1;
check_all_dirs "$images_dir" "$train_dir" || exit 1;

ctc_int="$(grep -w "<ctc>" "$syms_ctc" | awk '{print $2 + 1}')";

mkdir -p "$output_dir";
for p in te va; do
  lat="$output_dir/$p.lat";
  [ -s "$lat.ark" -a -s "$lat.scp" ] ||
  pylaia-htr-netout \
    --logging_also_to_stderr info \
    --logging_level info \
    --train_path "$train_dir" \
    --output_transform log_softmax \
    --output_format lattice \
    --show_progress_bar false \
    "$images_dir" \
    <(cut -d\  -f1 "data/kws_line/lang/char/$p.txt") |
    lattice-remove-ctc-blank "$ctc_int" ark:- ark:- |
    lattice-prune --beam="$lattice_beam" ark:- "ark,scp:$lat.ark,$lat.scp" ||
  exit 1;

  continue;
  : <<EOF
    lattice-prune-arcs --beam="$lattice_beam" ark:- ark:- |
  lattice-compose \
    ark:- \
    <(awk '$2 > 0{
             print 0, 1, $2 + 1, $2 + 1;
             print 1, 1, $2 + 1, $2 + 1;
           }END{
             print 1;
           }' "${syms_ctc}" | fstcompile) \
    "ark,scp:$lat.ark,$lat.scp" ||
  exit 1;

EOF
done;
