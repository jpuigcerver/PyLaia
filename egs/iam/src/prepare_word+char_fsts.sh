#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

boov="<unk>";
bos="<s>";
ctc="<ctc>";
eps="<eps>";
eps_on_replace=true;
eoov="</unk>";
eos="</s>";
loop_scale=1;
oov_penalty=0;
oov_scale=1.0;
oov_thresh=2;
overwrite=false;
ngram_order=5;
ngram_method=kneser_ney;
transition_scale=1;
help_message="
Usage: ${0##*/} [options] <syms> <char_gt> <output_dir>

Arguments:
  syms         : Table of symbols used for the CTC training.
  char_gt      : File with the char-transcript of each training sample.
  output_dir   : Output directory where the files will be written.

Options:
  --boov             : (type = string, value = \"$boov\")
                       Symbol used to represent the start of OOV words.
  --bos              : (type = string, value = \"$bos\")
                       Begin-of-sentence symbol.
  --ctc              : (type = string, value = \"$ctc\")
                       Symbol representing the CTC blank.
  --eps              : (type = string, value = \"$eps\")
                       Symbol representing the epsilon (no-symbol).
  --eps_on_replace   : (type = string, value = \"$eps_on_replace\")
                       If false, emit <boov> and <eoov> symbols when traversing
                       the OOV model.
  --eoov             : (type = string, value = \"$eoov\")
                       Symbol used to represent the end of OOV words.
  --eos              : (type = string, value = \"$eos\")
                       End-of-sentence symbol.
  --loop_scale       : (type = float, value = $loop_scale)
                       Scaling factor applied to the self-loops in the HMMs.
  --oov_penalty      : (type = float, value = $oov_penalty)
                       Penalty cost added to each OOV arc.
  --oov_scale        : (type = float, value = $oov_scale)
                       Scale factor applied to the OOV model.
  --oov_thresh       : (type = integer, value = $oov_thresh)
                       Consider OOV all words with less occurrencies than this.
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

# Get the word vocabulary, and split it into in-voc words and out-voc words.
voc="$(mktemp)";
cut -d\  -f2- "$2" | sort | uniq -c > "$voc";
awk -v t="$oov_thresh" '$1 >= t' "$voc" > "$voc.in";
awk -v t="$oov_thresh" '$1 < t' "$voc" > "$voc.out";

# Split transcriptions into in-voc and out-of-voc samples.
txt_in="$(mktemp)";
txt_out="$(mktemp)";
awk -v txt_in="$txt_in" -v txt_out="$txt_out" -v voc_in="$voc.in" '
BEGIN{
  while ((getline < voc_in) > 0) {
    w = $2;
    for (i = 3; i <= NF; ++i) { w=w" "$i; }
    VOC[w] = 1;
  }
}{
  w = $2;
  for (i = 3; i <= NF; ++i) { w=w" "$i; }
  if (w in VOC) {
    print $1, w > txt_in;
  } else {
    print $1, w > txt_out;
  }
}' "$2";

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
cut -d\  -f2- "$2" | sed -r 's|^ +||g;s| +$||g' |
awk -v bos="$bos" -v unk="$boov" -v voc="$voc.in" '
BEGIN{
  while ((getline < voc) > 0) {
    w = $2;
    for (i = 3; i <= NF; ++i) { w=w" "$i; }
    VOC[w] = 1;
  }

  TW = 0;
  print 0, 1, bos;
  N = 1;
}{
  if ($0 in VOC) { CW[$0]++; }
  else { CW[unk]++; }
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
fstarcsort --sort_type=ilabel > "$3/Gw.fst" ||
{ echo "ERROR: Creating Gw.fst!" >&2 && exit 1; }

# Create Gc transducer from out-vocabulary words (inc. oov token).
bos_int="$(grep -w "$bos" "$3/syms.txt" | awk '{print $2}')";
eos_int="$(grep -w "$eos" "$3/syms.txt" | awk '{print $2}')";
noov="$(wc -l "$txt_out" | awk '{print $1}')";
[[ "$overwrite" = false && -s "$3/Gc.fst" ]] ||
cut -d\  -f2- "$txt_out" |
farcompilestrings --generate_keys="$noov" \
		 --symbols="$3/syms.txt" --keep_symbols=false |
ngramcount --order="$ngram_order" --require_symbols=false |
ngrammake --method="$ngram_method" > "$3/Gc.fst" ||
{ echo "ERROR: Creating Gc.fst!" >&2 && exit 1; }

# Replace OOV ark in Gw, with the Gc.
# Note1: We can do this explicitly because there is only one OOV arc in Gw.
# If Gw was a word n-gram, we could not do this statically.
# Note2: We do not use Gc directly, but only accept paths with at least
# 1 character (see fstdifference).
boov_int="$(grep -w "$boov" "$3/syms.txt" | awk '{print $2}')";
eoov_int="$(grep -w "$eoov" "$3/syms.txt" | awk '{print $2}')";
boff_int="$(grep -w "#0" "$3/syms.txt" | awk '{print $2}')";
[[ "$overwrite" = false && -s "$3/G_p${oov_penalty}_s${oov_scale}.fst" ]] ||
fstreplace --call_arc_labeling=output --return_arc_labeling=output \
  	   --return_label="$eoov_int" --epsilon_on_replace="$eps_on_replace" \
	   <(fstprint "$3/Gw.fst" | \
	     awk -v boov="$boov_int" -v p="$oov_penalty" '{
               if (NF >= 4 && $3 == boov) { $5 = (NF == 4 ? p : $5 + p); }
               print;
             }' | fstcompile) \
	   -1 \
	   <(fstprint --acceptor "$3/Gc.fst" |
	     awk -v scale="$oov_scale" -v boff="$boff_int" '{
               if (NF == 4 || NF == 2) { $NF = $NF * scale; }
               if ($3 == 0) { $3 = boff; }
               print;
             }' | fstcompile --acceptor |
	     fstdifference - <(echo -e "0 0 $boff_int\n0" |
			       fstcompile --acceptor)) \
	   "$boov_int" |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstpushspecial |
fstrmsymbols <(echo "$boff_int") |
fstrmsymbols --apply-to-output=true <(echo "$boff_int") |
fstarcsort --sort_type=ilabel > "$3/G_p${oov_penalty}_s${oov_scale}.fst" ||
{ echo "ERROR: Creating G_p${oov_penalty}_s${oov_scale}.fst!" >&2 && exit 1; }
