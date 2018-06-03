#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

bos="<s>";
ctc="<ctc>";
eps="<eps>";
eos="</s>";
loop_scale=1;
overwrite=false;
transition_scale=1;
ngram_order=5;
ngram_method=kneser_ney;
help_message="
Usage: ${0##*/} [options] <syms> <char_gt> <output_dir>

Arguments:
  syms         : Table of symbols used for the CTC training.
  char_gt      :
  output_dir   :

Options:
  --bos              : (type = string, value = \"$bos\")
                       Begin-of-sentence symbol.
  --ctc              : (type = string, value = \"$ctc\")
                       Symbol representing the CTC blank.
  --eps              : (type = string, value = \"$eps\")
                       Symbol representing the epsilon (no-symbol).
  --eos              : (type = string, value = \"$eos\")
                       End-of-sentence symbol.
  --loop_scale       : (type = float, value = $loop_scale)
                       Scaling factor applied to the self-loops in the HMMs.
  --overwrite        : (type = boolean, value = $overwrite)
                       If true, existing files will be overwritten.
  --transition_scale : (type = float, value = $transition_scale)
                       Scaling factor applied to the outgoing arcs in the HMMs.
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

for f in "$1" "$2"; do
  [ ! -s "$f" ] &&
  echo "ERROR: File \"$f\" does not exist or is empty!" >&2 && exit 1;
done
mkdir -p "$3";

# Create "lexicon" file.
[[ "$overwrite" = false && -s "$3/lexiconp.txt" ]] ||
awk -v eps="$eps" -v ctc="$ctc" -v bos="$bos" -v eos="$eos" \
'$1 != eps && $1 != ctc{
  if ($1 == bos || $1 == eos) {
    print "Special symbol \""$1"\" found among the training symbols" \
      > "/dev/stderr";
    exit(1);
  }
  printf("%-5s 1.0 %s\n", $1, $1);
}END{
  printf("%-5s 1.0 %s\n", bos, ctc);
}' "$1" > "$3/lexiconp.txt";

# Add disambiguation symbols
tmp="$(mktemp)";
ndisambig=$("$SDIR/add_lex_disambig.pl" --pron-probs "$3/lexiconp.txt" "$tmp");
if [[ "$overwrite" = true || ! -s "$3/lexiconp_disambig.txt" ]] ||
     ! cmp -s "$tmp" "$3/lexiconp_disambig.txt"; then
  mv "$tmp" "$3/lexiconp_disambig.txt";
  overwrite=true;
fi;

# Symbols table
[[ "$overwrite" = false && -s "$3/syms.txt" ]] ||
awk -v eps="$eps" -v ctc="$ctc" -v bos="$bos" -v eos="$eos" \
    -v nd="$ndisambig" 'BEGIN{
  N = 0;
  printf("%-5s %d\n", eps, N++);
}$1 != eps{
  if ($1 == bos || $1 == eos) {
    print "Special symbol \""$1"\" found among the training symbols" \
      > "/dev/stderr";
    exit(1);
  }
  printf("%-5s %d\n", $1, N++);
}END{
  printf("%-5s %d\n", bos, N++);
  printf("%-5s %d\n", eos, N++);
  for (i=0; i <= nd; ++i) {
    printf("#%-4d %d\n", i, N++);
  }
}' "$1" > "$3/syms.txt";


# Create integer list of disambiguation symbols.
gawk '$1 ~ /^#.+/{ print $2 }' "$3/syms.txt" > "$3/syms_disambig.int";

# Create HMM model and tree
"$SDIR/create_ctc_hmm_model.sh" \
  --dregex "^(#.+|<s>|</s>)" \
  --eps "$eps" --ctc "$ctc" --overwrite "$overwrite" \
  "$3/syms.txt" "$3/model" "$3/tree";

# Create the lexicon FST with disambiguation symbols.
[[ "$overwrite" = false && -s "$3/L.fst" ]] ||
"$SDIR/make_lexicon_fst.pl" --pron-probs "$3/lexiconp_disambig.txt" |
fstcompile --isymbols="$3/syms.txt" --osymbols="$3/syms.txt" |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstarcsort --sort_type=ilabel > "$3/L.fst" ||
{ echo "Failed $3/L.fst creation!" >&2 && exit 1; }

# Compose the context-dependent and the L transducers.
[[ "$overwrite" = false && -s "$3/CL.fst" ]] ||
fstcomposecontext --context-size=1 --central-position=0 \
		  --read-disambig-syms="$3/syms_disambig.int" \
		  --write-disambig-syms="$3/ilabels_disambig.int" \
		  "$3/ilabels" "$3/L.fst" |
fstarcsort --sort_type=ilabel > "$3/CL.fst" ||
{ echo "Failed $3/CL.fst creation!" >&2 && exit 1; }

# Create Ha transducer
[[ "$overwrite" = false && -s "$3/Ha.fst" ]] ||
make-h-transducer \
    --disambig-syms-out="$3/tid_disambig.int" \
    --transition-scale="$transition_scale" \
    "$3/ilabels" "$3/tree" "$3/model" > "$3/Ha.fst" ||
{ echo "Failed $3/Ha.fst creation!" >&2 && exit 1; }

# Create HaCL transducer.
[[ "$overwrite" = false && -s "$3/HCL.fst" ]] ||
fsttablecompose "$3/Ha.fst" "$3/CL.fst" |
fstdeterminizestar --use-log=true |
fstrmsymbols "$3/tid_disambig.int" |
fstrmepslocal |
fstminimizeencoded > "$3/HaCL.fst" ||
{ echo "Failed $3/HaCL.fst creation!" >&2 && exit 1; }

# Create HCL transducer.
[[ "$overwrite" = false && -s "$3/HCL.fst" ]] ||
add-self-loops --self-loop-scale="$loop_scale" --reorder=true \
	       "$3/model" "$3/HaCL.fst" |
fstarcsort --sort_type=olabel > "$3/HCL.fst" ||
{ echo "Failed $3/HCL.fst creation!" >&2 && exit 1; }

# Create G transducer.
bos_int="$(grep -w "$bos" "$3/syms.txt" | awk '{print $NF}')";
eos_int="$(grep -w "$eos" "$3/syms.txt" | awk '{print $NF}')";
boff_int="$(grep -w "#0" "$3/syms.txt" | awk '{print $NF}')";
cut -d\  -f2- "$2" > "$tmp";
[[ "$overwrite" = false && -s "$3/G.fst" ]] ||
farcompilestrings --symbols="$3/syms.txt" --keep_symbols=false "$tmp" |
ngramcount --order="$ngram_order" --require_symbols=false  |
ngrammake --method="$ngram_method" |
fstdifference - <(echo "0" | fstcompile) | fstprint |
awk -v boff="$boff_int" '{ if ($3 == 0) $3 = boff; print; }' | fstcompile |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstrmsymbols <(echo "$boff_int") |
fstconcat <(echo -e "0 1 $bos_int\n1" | fstcompile --acceptor) - |
fstarcsort --sort_type=ilabel > "$3/G.fst" ||
{ echo "ERROR: Creating G.fst!" >&2 && exit 1; }

rm "$tmp";
exit 0;
