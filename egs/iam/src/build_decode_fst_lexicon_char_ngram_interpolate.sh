#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
export PATH="$SDIR/../../utils:$PATH";

bos="<s>";
ctc="<ctc>";
eps="<eps>";
eos="</s>";
loop_scale=1;
ngram_method=kneser_ney;
ngram_order=5;
ngram_renormalize=true;
oov_penalty=0;
oov_probability=0.102855;
overwrite=false;
push_weights=true;
transition_scale=1;
help_message="
Usage: ${0##*/} [options] <syms> <char_gt> <output_dir>

Arguments:
  syms         : Table of symbols used for the CTC training.
  char_gt      : File with the char-transcript of each training sample.
  output_dir   : Output directory where the files will be written.

Options:
  --bos               : (type = string, value = \"$bos\")
                        Begin-of-sentence symbol.
  --ctc               : (type = string, value = \"$ctc\")
                        Symbol representing the CTC blank.
  --eps               : (type = string, value = \"$eps\")
                        Symbol representing the epsilon (no-symbol).
  --eos               : (type = string, value = \"$eos\")
                        End-of-sentence symbol.
  --loop_scale        : (type = float, value = $loop_scale)
                        Scaling factor applied to the self-loops in the HMMs.
  --ngram_method      : (type = string, value = $ngram_method)
                        Discounting method for n-grams (see ngrammake --help).
  --ngram_order       : (type = integer, value = $ngram_order)
                        Order of the n-gram model.
  --ngram_renormalize : (type = boolean, value = $ngram_renormalize)
                        If true, re-normalize the character n-gram used to
                        model out-of-vocabulary words. CAUTION: the n-gram fst
                        must encode a finite weight mass.
  --oov_penalty       : (type = float, value = $oov_penalty)
                        Add this constant penalty to the out-of-vocabulary FST.
  --oov_probability   : (type = float, value = $oov_probability)
                        Estimated probability for of OOV words.
  --overwrite         : (type = boolean, value = $overwrite)
                        If true, existing files will be overwritten.
  --push_weights      : (type = boolean, value = $push_weights)
                        If true, push weights of the final grammar FST.
  --transition_scale  : (type = float, value = $transition_scale)
                        Scaling factor applied to the outgoing arcs in the HMMs.
";
source "$SDIR/../../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

for f in "$1" "$2"; do
  [ ! -s "$f" ] &&
  echo "ERROR: File \"$f\" does not exist or is empty!" >&2 && exit 1;
done
mkdir -p "$3";

[[ "$overwrite" = false && -s "$3/lexiconp.txt" ]] ||
awk -v bos="$bos" -v ctc="$ctc" '
$1 != ctc{
  printf("%-5s %f %s\n", $1, 1.0, $1);
}END{
  printf("%-5s %f %s\n", bos, 1.0, ctc);
}' "$1" > "$3/lexiconp.txt";

# Add disambiguation symbols
tmp="$(mktemp)";
ndisambig=$(add_lex_disambig.pl --pron-probs "$3/lexiconp.txt" "$tmp");
if [[ "$overwrite" = true || ! -s "$3/lexiconp_disambig.txt" ]] ||
     ! cmp -s "$tmp" "$3/lexiconp_disambig.txt"; then
  mv "$tmp" "$3/lexiconp_disambig.txt";
  overwrite=true;
fi;

# Character symbols
[[ "$overwrite" = false && -s "$3/syms.txt" ]] ||
sort -nk2 "$1" |
awk -v eps="$eps" -v bos="$bos" -v eos="$eos" \
    -v nd="$ndisambig" '
BEGIN{
  N = 0;
  printf("%-10s %d\n", eps, N++);
}{
  printf("%-10s %d\n", $1, N++);
}END{
  printf("%-10s %d\n", bos, N++);
  printf("%-10s %d\n", eos, N++);
  for (d = 0; d <= nd; ++d) {
    printf("#%-9d %d\n", d, N++);
  }
}' > "$3/syms.txt";

# Create integer list of disambiguation symbols.
gawk '$1 ~ /^#.+/{ print $2 }' "$3/syms.txt" > "$3/syms_disambig.int";

# Create HMM model and tree
create_ctc_hmm_model.sh \
  --eps "$eps" --ctc "$ctc" --overwrite "$overwrite" \
  --dregex "^(#.+|$bos|$eos)" \
  "$3/syms.txt" "$3/model" "$3/tree";

# Create the lexicon FST with disambiguation symbols.
[[ "$overwrite" = false && -s "$3/L.fst" ]] ||
make_lexicon_fst.pl --pron-probs "$3/lexiconp_disambig.txt" |
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
fstminimizeencoded |
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

# Create Gw transducer from lexicon words
[[ "$overwrite" = false && -s "$3/Gw.fst" ]] ||
cut -d\  -f2- "$2" | sed -r 's|^ +||g' |
awk 'BEGIN{
  TW = 0;
}{
  CW[$0]++;
  TW++;
}END{
  N = 0;
  for (w in CW) {
    n = split(w, aw, " ");
    print 0, ++N, aw[1], log(TW) - log(CW[w]);
    for (i = 2; i <= n; ++i) {
      printf("%d %d %s\n", N, ++N, aw[i]);
    }
    print N;
  }
}' |
fstcompile --acceptor --isymbols="$3/syms.txt" |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstpushspecial |
fstarcsort --sort_type=ilabel > "$3/Gw.fst" ||
{ echo "ERROR: Creating Gw.fst!" >&2 && exit 1; }

# Create an unweighted version of Gw.
[[ "$overwrite" = false && -s "$3/Gw.fsa" ]] ||
fstprint --acceptor "$3/Gw.fst" |
awk '{
 if (NF == 2) { print $1; }
 else if (NF == 4) { print $1, $2, $3; }
 else { print; }
}' | fstcompile --acceptor | fstdeterminize > "$3/Gw.fsa" ||
{ echo "ERROR: Creating Gw.fsa!" >&2 && exit 1; }


if [ -n "$oov_probability" ]; then
  oov_penalty=$(echo "-l($oov_probability) + $oov_penalty" | bc -l);
  inv_penalty=$(echo "-l(1.0 - $oov_probability)" | bc -l);
else
  inv_penalty=0;
fi;


# Create Gc transducer from lexicon words.
cut -d\  -f2- "$2" | sed -r 's|^ +||g' > "$tmp";
[[ "$overwrite" = false && -s "$3/Gc.fst" ]] ||
farcompilestrings --symbols="$3/syms.txt" --keep_symbols=false "$tmp" |
ngramcount --order="$ngram_order" --require_symbols=false  |
ngrammake --method="$ngram_method" |
fstdifference - <(echo "0" | fstcompile) |
fstarcsort --sort_type=ilabel > "$3/Gc.fst" ||
{ echo "ERROR: Creating Gcw2.fst!" >&2 && exit 1; }


# Create the FST to represent in-vocabulary words.
# We add the mass from original word-lexicon FST and the mass from
# the n-gram corresponding to these words.
[[ "$overwrite" = false && -s "$3/Gcw1.fst" ]] || (
  set -e;
  fstcompose <(fstarcsort --sort_type=olabel "$3/Gc.fst") "$3/Gw.fsa" |
  fstunion "$3/Gw.fst" - > "$tmp";
  offset=$(fstprint "$tmp" | fstcompile --arc_type=log |
           fstshortestdistance --reverse=true | head -n1 |
	   awk '{print -1 * $NF}');
  offset=$(echo "$inv_penalty + $offset" | bc -l);
  fstconcat <(echo -e "0 1 0 0 $offset\n1" | fstcompile) "$tmp" > "$3/Gcw1.fst";
) || { echo "ERROR: Creating Gcw1.fst!" >&2 && exit 1; }


# Create the FST to represent out-of-vocabulary words.
# We subtract the in-vocabulary words from the valid paths in the original
# character n-gram FST, and redistribute the probability mass.
boff_int="$(grep -w "#0" "$3/syms.txt" | awk '{print $NF}')";
[[ "$overwrite" = false && -s "$3/Gcw2.fst" ]] || (
  set -e;
  fstdifference "$3/Gc.fst" "$3/Gw.fsa" > "$tmp";
  if [[ "$ngram_renormalize" = true ]]; then
    echo "WARNING: You are trying to re-normalize the n-gram FST." \
	 "If this takes too long, it's probably because the FST does" \
	 "not have a finite mass. Use \"--ngram_renormalize false\"." >&2;
    offset=$(fstprint "$tmp" | fstcompile --arc_type=log |
             fstshortestdistance --reverse=true | head -n1 |
	     awk '{print -1 * $NF}');
  else
    offset=0;
  fi;
  offset=$(echo "$oov_penalty + $offset" | bc -l);
  # Note: We add disambiguation symbols to the backoff states here.
  fstconcat <(echo -e "0 1 0 0 $offset\n1" | fstcompile) \
	    <(fstprint "$tmp" |
              awk -v boff="$boff_int" '{ if ($3 == 0) $3 = boff; print; }' |
	      fstcompile) \
	    > "$3/Gcw2.fst";
) || { echo "ERROR: Creating Gcw2.fst!" >&2 && exit 1; }

# Union of the two fsts. Gcw2 is unambiguous because of the disambiguation
# symbols to the backoff states. We can determinize the union fst to properly
# sum the proabilities of repeated paths.
bos_int="$(grep -w "$bos" "$3/syms.txt" | awk '{print $NF}')";
fstunion "$3/Gcw1.fst" "$3/Gcw2.fst" |
fstconcat <(echo -e "0 1 $bos_int\n1" | fstcompile --acceptor) - |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
if [ "$push_weights" = true ]; then fstpushspecial; else cat; fi |
fstproject --project_output=true |
fstarcsort --sort_type=ilabel > "$3/G.fst" ||
{ echo "ERROR: Creating G.fst!" >&2 && exit 1; }

exit 0;
