#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export PYTHONPATH="$PWD/../..:$PYTHONPATH";
export PATH="$PWD/../utils:$PATH";
export PATH="$PWD/../..:$PATH";

# Check for required files
check_all_files -s data/bentham/lang/syms_ctc.txt \
                   data/bentham/lang/char/te.txt \
	               data/bentham/lang/char/tr.txt \
	               data/bentham/lang/char/va.txt \
	               data/bentham/train/experiment.ckpt-90;

# Parse options
beam=15;
help_message="
Usage: ${0##*/} [options]

Options:
  --beam : (type = float, default = $beam)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;


mkdir -p data/bentham/lats/ctc_lb${beam};
for p in te va; do
  lat=data/bentham/lats/ctc_lb${beam}/${p}.lat;
  pylaia-htr-netout \
    --logging_also_to_stderr info \
    --logging_level info \
    --train_path data/bentham/train \
    --output_transform log_softmax \
    --output_format lattice \
    data/bentham/imgs/lines_h80 \
    <(cut -d\  -f1 data/bentham/lang/char/$p.txt) |
  lattice-prune --beam="$beam" ark:- "ark,scp:$lat.ark,$lat.scp";
done;
