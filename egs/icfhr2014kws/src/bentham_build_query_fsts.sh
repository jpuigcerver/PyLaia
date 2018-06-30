#!/usr/bin/env bash
set -e;
export LC_ALL=en_US.utf8;
export PATH="$PWD/../utils:$PATH";
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

loop_scale=1;
ngram_method=kneser_ney;
ngram_order=7;
overwrite=false;
transition_scale=1;
help_message="
Usage: ${0##*/} <output_dir>

Important:
  This recipe assumes that each word starts with the whitespace symbol \"<sp>\",
  you should add a \"fake\" frame at the start of your likelihood matrices with
  all the mass assigned to that symbol.

Options:
  --loop_scale
  --ngram_method
  --ngram_order
  --overwrite
  --transition_scale
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] &&
echo -e "ERROR: Missing arguments!\n$help_message" >&2 &&
exit 1;

# Create output dir
mkdir -p "$1";

# Create the lexicon file
[ "$overwrite" = false -a -s "$1/lexicon.txt" ] ||
gawk '{
  if ($1 != "<eps>" && $1 != "<ctc>") {
    printf("%8-s %s\n", $1, $1);
  }
}' data/bentham/lang/syms_ctc.txt > "$1/lexicon.txt";


# Create char symbols, including disambiguation symbols
[ "$overwrite" = false -a \
  -s "$1/lexicon_disambig.txt" -a -s "$1/chars.txt" ] || {
  ndisambig=$(add_lex_disambig.pl "$1/lexicon.txt" "$1/lexicon_disambig.txt");
  gawk -v nd="$ndisambig" 'BEGIN{
    print "<eps>", 0;
  }{
    print $1, NR;
  }END{
    for (d = 0; d <= nd; ++d) {
      printf("#%d %d\n", d, NR + d + 1);
    }
  }' data/bentham/lang/syms_ctc.txt > "$1/chars.txt";
}

# Create integer list of disambiguation symbols.
[ "$overwrite" = false -a -s "$1/chars_disambig.int" ] ||
gawk '$1 ~ /^#[0-9]+/{ print $2 }' "$1/chars.txt" > "$1/chars_disambig.int";
char_disambig_sym=`grep \#0 "$1/chars.txt" | awk '{print $2}'`;

# Create alignment lexicon (to use with lattice-align-words-lexicon)
[ "$overwrite" = false -a -s "$1/lexicon_align.txt" ] ||
awk '{
  printf("%10-s %10-s", $1, $1);
  for (i = 2; i <= NF; ++i) { printf(" %s", $i); }
  printf("\n");
}' "$1/lexicon.txt" |
sym2int.pl -f 1- "$1/chars.txt" > "$1/lexicon_align.txt";


# Create HMM model and tree
create_ctc_hmm_model.sh \
  --eps "<eps>" --ctc "<ctc>" \
  --overwrite "$overwrite" \
  "$1/chars.txt" \
  "$1/model" \
  "$1/tree";

# Create n-gram language model
[ "$overwrite" = false -a -s "$1/lm.fst" ] || {
  tmp="$(mktemp)";
  cut -d\  -f2- data/bentham/lang/char/tr.txt | sed -r 's|<sp>|\n|g' > "$tmp";
  farcompilestrings --symbols="$1/chars.txt" --keep_symbols=true "$tmp" |
  ngramcount --order="$ngram_order" --require_symbols=true |
  ngrammake --method="$ngram_method" > "$1/lm.fst";
  rm "$tmp";
}

# Compute perplexity
tmp="$(mktemp)";
for p in va te; do
  echo "=== $p ===";
  cut -d\  -f2- "data/bentham/lang/char/$p.txt" | sed -r 's|<sp>|\n|g' > "$tmp";
  farcompilestrings --symbols="$1/chars.txt" --keep_symbols=true "$tmp" |
  ngramperplexity "$1/lm.fst" 2>&1 |
  grep -v "WARNING: OOV probability" |
  grep -v "NOTE:";
done;
rm "$tmp";

# Create L with disambiguation symbols.
# Self-loops are added to propagate the backoff arcs (#0) from the
# language model (see next).
[[ "$overwrite" = false && -s "$1/L.fst" ]] ||
make_lexicon_fst.pl "$1/lexicon_disambig.txt" |
fstcompile --isymbols="$1/chars.txt" --osymbols="$1/chars.txt" \
     --keep_isymbols=false --keep_osymbols=false |
fstaddselfloops "echo $char_disambig_sym |" "echo $char_disambig_sym |" | \
fstarcsort --sort_type=olabel > "$1/L.fst";

# Compose LG with disambiguation symbols.
# We need the disambiguation symbols because we are going to determinize
# the resulting FST.
# We enforce that text starts with the character <sp> to handle the fake
# frame added at the start of the utterance.
sp_int="$(grep -w "<sp>" $1/chars.txt | awk '{ print $2 }')";
[[ "$overwrite" = false && -s "$1/LG.fst" ]] ||
fstrelabel \
--relabel_ipairs=<(echo "0 $char_disambig_sym") \
"$1/lm.fst" |
fstconcat <(echo -e "0  1  $sp_int $sp_int\n0" | fstcompile) - |
fstarcsort --sort_type=ilabel |
fsttablecompose "$1/L.fst" - |
fstdeterminizestar --use-log=true |
fstminimizeencoded | \
fstpushspecial |
fstarcsort --sort_type=ilabel > "$1/LG.fst";

# Compose the context-dependent and the L transducers.
[[ "$overwrite" = false && -s "$1/CLG.fst" ]] ||
fstcomposecontext \
--context-size=1 \
--central-position=0 \
--read-disambig-syms="$1/chars_disambig.int" \
--write-disambig-syms="$1/ilabels_disambig.int" \
"$1/ilabels" \
"$1/LG.fst" |
fstarcsort --sort_type=ilabel > "$1/CLG.fst" ||
{ echo "Failed $1/CLG.fst creation!" >&2 && exit 1; }


# Create Ha transducer
[[ "$overwrite" = false && -s "$1/Ha.fst" ]] ||
make-h-transducer \
--disambig-syms-out="$1/tid_disambig.int" \
--transition-scale="$transition_scale" \
"$1/ilabels" \
"$1/tree" \
"$1/model" > "$1/Ha.fst" ||
{ echo "Failed $1/Ha.fst creation!" >&2 && exit 1; }

# Create HaCLG transducer.
# Note: This is the HCLG transducer without self-loops.
[[ "$overwrite" = false && -s "$1/HaCLG.fst" ]] ||
fsttablecompose "$1/Ha.fst" "$1/CLG.fst" | \
fstdeterminizestar --use-log=true | \
fstrmsymbols "$1/tid_disambig.int" | \
fstrmepslocal  |
fstminimizeencoded > "$1/HaCLG.fst" ||
{ echo "Failed $1/HaCLG.fst creation!" >&2 && exit 1; }


# Create HCLG transducer.
[[ "$overwrite" = false && -s "$1/HCLG.fst" ]] ||
add-self-loops \
--self-loop-scale="$loop_scale" \
--reorder=true \
"$1/model" "$1/HaCLG.fst" |
fstarcsort --sort_type=olabel > "$1/HCLG.fst" ||
{ echo "Failed $1/HCLG.fst creation!" >&2 && exit 1; }