#!/usr/bin/env bash
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/../..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";

ctc="<ctc>";
eps="<eps>";
loop_scale=1;
overwrite=false;
transition_scale=1;
help_message="
Usage: ${0##*/} [options] lm_dir output_dir

Options:
  --ctc              : (type = string, default = \"$ctc\")
  --eps              : (type = string, default = \"$eps\")
  --loop_scale       : (type = float, default = $loop_scale)
  --overwrite        : (type = boolean, default = $overwrite)
  --transition_scale : (type = float, default = $transition_scale)
";
source $PWD/../utils/parse_options.inc.sh || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

lm_dir="$1";
output_dir="$2";
mkdir -p "$output_dir" || exit 1;
check_all_dirs "$lm_dir" || exit 1;
check_all_files -s "$lm_dir/chars.txt" "$lm_dir/words.txt" \
                "$lm_dir/lexicon_disambig.txt" "$lm_dir/lm.fst" || exit 1;

# Create integer list of disambiguation symbols.
[ "$overwrite" = false -a -s "$output_dir/chars_disambig.int" ] ||
gawk '$1 ~ /^#.+/{ print $2 }' "$lm_dir/chars.txt" > "$output_dir/chars_disambig.int";
# Create integer list of disambiguation symbols.
[ "$overwrite" = false -a -s "$output_dir/words_disambig.int" ] ||
gawk '$1 ~ /^#.+/{ print $2 }' "$lm_dir/words.txt" > "$output_dir/words_disambig.int";

char_disambig_sym=`grep \#0 "$lm_dir/chars.txt" | awk '{print $2}'`;
word_disambig_sym=`grep \#0 "$lm_dir/words.txt" | awk '{print $2}'`;

# Create HMM model and tree
create_ctc_hmm_model.sh \
  --eps "$eps" --ctc "$ctc" \
  --overwrite "$overwrite" \
  "$lm_dir/chars.txt" \
  "$output_dir/model" \
  "$output_dir/tree";

# Create L with disambiguation symbols.
# Self-loops are added to propagate the backoff arcs (#0) from the
# language model (see next).
[[ "$overwrite" = false && -s "$output_dir/L.fst" ]] ||
make_lexicon_fst.pl "$lm_dir/lexicon_disambig.txt" |
fstcompile --isymbols="$lm_dir/chars.txt" --osymbols="$lm_dir/words.txt" \
           --keep_isymbols=false --keep_osymbols=false |
fstaddselfloops "echo $char_disambig_sym |" "echo $word_disambig_sym |" | \
fstarcsort --sort_type=olabel > "$output_dir/L.fst" || exit 1;

# Compose LG with disambiguation symbols.
# We need the disambiguation symbols because we are going to determinize
# the resulting FST.
[[ "$overwrite" = false && -s "$output_dir/LG.fst" ]] ||
fstrelabel --relabel_ipairs=<(echo "0 $word_disambig_sym") "$lm_dir/lm.fst" |
fstarcsort --sort_type=ilabel |
fsttablecompose "$output_dir/L.fst" - |
fstdeterminizestar --use-log=true |
fstminimizeencoded | \
fstpushspecial |
fstarcsort --sort_type=ilabel > "$output_dir/LG.fst";

# Compose the context-dependent and the L transducers.
[[ "$overwrite" = false && -s "$output_dir/CLG.fst" ]] ||
fstcomposecontext \
  --context-size=1 \
  --central-position=0 \
  --read-disambig-syms="$lm_dir/chars_disambig.int" \
  --write-disambig-syms="$lm_dir/ilabels_disambig.int" \
  "$output_dir/ilabels" \
  "$output_dir/LG.fst" |
fstarcsort --sort_type=ilabel > "$output_dir/CLG.fst" ||
{ echo "Failed $output_dir/CLG.fst creation!" >&2 && exit 1; }

# Create Ha transducer
[[ "$overwrite" = false && -s "$output_dir/Ha.fst" ]] ||
make-h-transducer \
  --disambig-syms-out="$output_dir/tid_disambig.int" \
  --transition-scale="$transition_scale" \
  "$output_dir/ilabels" \
  "$output_dir/tree" \
  "$output_dir/model" > "$output_dir/Ha.fst" ||
{ echo "Failed $output_dir/Ha.fst creation!" >&2 && exit 1; }

# Create HaCLG transducer.
# Note: This is the HCLG transducer without self-loops.
[[ "$overwrite" = false && -s "$output_dir/HaCLG.fst" ]] ||
fsttablecompose "$output_dir/Ha.fst" "$output_dir/CLG.fst" | \
fstdeterminizestar --use-log=true | \
fstrmsymbols "$output_dir/tid_disambig.int" | \
fstrmepslocal  |
fstminimizeencoded > "$output_dir/HaCLG.fst" ||
{ echo "Failed $output_dir/HaCLG.fst creation!" >&2 && exit 1; }

# Create HCLG transducer.
[[ "$overwrite" = false && -s "$output_dir/HCLG.fst" ]] ||
add-self-loops \
  --self-loop-scale="$loop_scale" \
  --reorder=true \
  "$output_dir/model" "$output_dir/HaCLG.fst" |
fstarcsort --sort_type=olabel > "$output_dir/HCLG.fst" ||
{ echo "Failed $output_dir/HCLG.fst creation!" >&2 && exit 1; }
