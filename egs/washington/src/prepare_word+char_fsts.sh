#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

boov="<unk>";
bos="<s>";
ctc="<ctc>";
eps="<eps>";
eoov="</unk>";
eos="</s>";
loop_scale=1;
oov_prob=0.1563785;
oov_scale=1;
overwrite=false;
srilm_opts="-order 6 -wbdiscount -interpolate";
transition_scale=1;
help_message="
Usage: ${0##*/} [options] <syms> <char_gt> <output_dir>

Arguments:
  syms         : Table of symbols used for the CTC training.
  char_gt      : File with the char-transcript of each training sample.
  word_gt      : File with the word-transcript of each training sample.
  output_dir   : Output directory where the files will be written.

Options:
  --boov             : (type = string, value = \"$boov\")
                       Symbol used to represent the start of out-of-vocabulary words.
  --bos              : (type = string, value = \"$bos\")
                       Begin-of-sentence symbol.
  --ctc              : (type = string, value = \"$ctc\")
                       Symbol representing the CTC blank.
  --eps              : (type = string, value = \"$eps\")
                       Symbol representing the epsilon (no-symbol).
  --eoov             : (type = string, value = \"$eoov\")
                       Symbol used to represent the end of out-of-vocabulary words.
  --eos              : (type = string, value = \"$eos\")
                       End-of-sentence symbol.
  --loop_scale       : (type = float, value = $loop_scale)
                       Scaling factor applied to the self-loops in the HMMs.
  --oov_prob         : (type = float, value = $oov_prob)
                       Estimated frequency of OOV words.
  --oov_scale        : (type = float, value = $oov_scale)
                       Apply this scale factor to the weights of the OOV FST.
  --overwrite        : (type = boolean, value = $overwrite)
                       If true, existing files will be overwritten.
  --srilm_opts       : (type = string, value = \"$srilm_opts\")
                       SRILM options for building the character n-gram.
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

[[ "$overwrite" = false && -s "$3/lexiconp.txt" ]] ||
awk -v bos="$bos" -v ctc="$ctc" '
$1 != ctc{
  printf("%-5s %f %s\n", $1, 1.0, $1);
}END{
  printf("%-5s %f %s\n", bos, 1.0, ctc);
}' "$1" > "$3/lexiconp.txt";

# Add disambiguation symbols
tmp="$(mktemp)";
ndisambig=$("$SDIR/add_lex_disambig.pl" --pron-probs "$3/lexiconp.txt" "$tmp");
if [[ "$overwrite" = true || ! -s "$3/lexiconp_disambig.txt" ]] ||
     ! cmp -s "$tmp" "$3/lexiconp_disambig.txt"; then
  mv "$tmp" "$3/lexiconp_disambig.txt";
  overwrite=true;
fi;

# Character symbols
[[ "$overwrite" = false && -s "$3/syms.txt" ]] ||
sort -nk2 "$1" |
awk -v eps="$eps" -v bos="$bos" -v eos="$eos" -v boov="$boov" -v eoov="$eoov" \
    -v nd="$ndisambig" '
BEGIN{
  N = 0;
  printf("%-10s %d\n", eps, N++);
}{
  printf("%-10s %d\n", $1, N++);
}END{
  printf("%-10s %d\n", bos, N++);
  printf("%-10s %d\n", eos, N++);
  printf("%-10s %d\n", boov, N++);
  printf("%-10s %d\n", eoov, N++);
  for (d = 0; d <= nd; ++d) {
    printf("#%-9d %d\n", d, N++);
  }
}' > "$3/syms.txt";

# Create integer list of disambiguation symbols.
gawk '$1 ~ /^#.+/{ print $2 }' "$3/syms.txt" > "$3/syms_disambig.int";

# Create HMM model and tree
"$SDIR/create_ctc_hmm_model.sh" \
  --eps "$eps" --ctc "$ctc" --overwrite "$overwrite" \
  --dregex "^(#.+|$bos|$eos|$boov|$eoov)" \
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

# Create Gw transducer from words (inc. oov token).
[[ "$overwrite" = false && -s "$3/Gw.fst" ]] ||
cut -d\  -f2- "$2" | sed -r 's|^ +||g' |
awk -v bos="$bos" '
BEGIN{
  TW = 0;
  print 0, 1, bos;
  N = 1;
}{
  CW[$0]++;
  TW++;
}END{
  for (w in CW) {
    n = split(w, aw, " ");
    print 1, ++N, aw[1], log(TW) - log(CW[w]);
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
fstarcsort --sort_type=ilabel > "$3/Gw.fst";

# Create Gc transducer from out-vocabulary words (inc. oov token).
bos_int="$(grep -w "$bos" "$3/syms.txt" | awk '{print $NF}')";
eos_int="$(grep -w "$eos" "$3/syms.txt" | awk '{print $NF}')";
backoff_int="$(grep -w "#0" "$3/syms.txt" | awk '{print $NF}')";
[[ "$overwrite" = false && -s "$3/Gc.fst" ]] ||
cut -d\  -f2- "$2" | sed -r 's|^ +||g' |
ngram-count $srilm_opts -text - -lm - |
arpa2fst --bos-symbol="$bos" --eos-symbol="$eos" --keep-symbols=false \
	 --disambig-symbol="#0" --read-symbol-table="$3/syms.txt" /dev/stdin |
fstconcat <(echo -e "0 1 $bos_int\n1" | fstcompile --acceptor) - |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstproject --project_output=false |
fstarcsort --sort_type=ilabel > "$3/Gc.fst";

# Create an unweighted version of Gw, which also include the special <s> word.
[[ "$overwrite" = false && -s "$3/Gw.unweight.fst" ]] ||
fstprint --acceptor "$3/Gw.fst" |
awk -v bos="$bos_int" '{
 if (NF == 2) { print $1; }
 else if (NF == 4) { print $1, $2, $3; }
 else { print; }
}END{
  printf("%d %d %d\n", 0, 999999999, bos);  # Remove empty string from Gc
  printf("%d\n", 999999999);
}' | fstcompile --acceptor | fstdeterminize > "$3/Gw.unweighted.fst";

# Compute the probability mass corresponding to the in-vocabulary words in the
# char-level n-gram. We will use this to approximate what's the probability
# mass left in Gc - Gw, since we cannot compute this directly due to the
# backoff loops.
# Note: MAKE SURE #0 are not removed from Gc
#iv_prob=$(fstcompose "$3/Gc.fst" "$3/Gw.unweighted.fst" | fstprint |
#	  fstcompile --arc_type=log |
#	  fstshortestdistance --reverse=true | head -n1 | awk '{print $2}');

# Remove the paths in Gw from Gc, without taking into consideration the backoff
# states.
[[ "$overwrite" = false && -s "$3/Gcmw.fst" ]] ||
fstdifference "$3/Gc.fst" "$3/Gw.unweighted.fst" > "$3/Gcmw.fst";

# TODO: We are not taking care of the lost mass in any way, since the iv_prob
# approximation is not very accurate, especially for large ngram orders, where
# it seems that the mass is unbound.
#for d in 0.01 0.001 0.0001 0.00001; do
#  fstprint "$3/Gcmw.fst" | fstcompile --arc_type=log |
# fstshortestdistance --delta=$d --reverse=true | head -n1 | awk '{print $2}';
#done;

[[ "$overwrite" = false && -s "$3/G.fst" ]] ||
fstunion <(fstprint --acceptor "$3/Gw.fst" |
	   awk -v bos="$bos_int" -v p="$oov_prob" '
           {
             if ($3 == bos) {
               print $1, $2, $3, $4 - log(1.0 - p);
             } else {
               print;
             }
           }' | fstcompile --acceptor) \
	 <(fstprint --acceptor "$3/Gcmw.fst" |
	   awk -v bos="$bos_int" -v p="$oov_prob" -v s="$oov_scale" '
           {
             if ($3 == bos) {
               print $1, $2, $3, $4 - log(p);
             } else {
               if (NF == 4 || NF == 2) $NF = $NF * s;
               print;
             }
           }' | fstcompile --acceptor) |
fstrmepslocal |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstrmsymbols <(echo "$backoff_int") |
fstproject --project_output=false |
fstrmepslocal |
fstarcsort --sort_type=ilabel > "$3/G.fst";
