#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

boov="<unk>";
bos="<s>";
char_prefix="*";
ctc="<ctc>";
eps="<eps>";
eoov="</unk>";
eos="</s>";
loop_scale=1;
oov_penalty=0;
oov_thresh=2;
overwrite=false;
srilm_opts="-order 7 -wbdiscount";
transition_scale=1;
help_message="
Usage: ${0##*/} [options] <syms> <char_gt> <word_gt> <output_dir>

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
  --char_prefix      : (type = string, value = \"$char_prefix\")
                       Add this prefix to the characters, to use them in the
                       combined Char+Word LM.
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
[ $# -ne 4 ] && echo "$help_message" >&2 && exit 1;

for f in "$1" "$2" "$3"; do
  [ ! -s "$f" ] &&
  echo "ERROR: File \"$f\" does not exist or is empty!" >&2 && exit 1;
done
mkdir -p "$4";

# Get the word vocabulary, and split it into in-voc words and out-voc words.
voc="$(mktemp)";
cut -d\  -f2 "$3" | sort | uniq -c > "$voc";
awk -v t="$oov_thresh" '$1 >= t' "$voc" > "$voc.in";
awk -v t="$oov_thresh" '$1 < t' "$voc" > "$voc.out";

# Get transcriptions of the in-vocabulary samples.
wtxt_in="$(mktemp)";
join -1 2 <(sort -k2 "$3") <(awk '{print $2}' "$voc.in" | sort) |
awk '{print $2, $1}' | sort > "$wtxt_in";
ctxt_in="$(mktemp)";
join -1 1 <(cut -d\  -f1 "$wtxt_in") <(sort "$2") | sort > "$ctxt_in";

# Get transcriptions of the out-of-vocabulary samples.
wtxt_out="$(mktemp)";
comm -23 <(sort "$3") "$wtxt_in" | sort > "$wtxt_out";
ctxt_out="$(mktemp)";
comm -23 <(sort "$2") "$ctxt_in" | sort > "$ctxt_out";


[[ "$overwrite" = false && -s "$4/lexiconp.txt" ]] || (
  # Lexicon contains the in-vocabulary words
  join -1 1 "$wtxt_in" "$ctxt_in" |
  awk -v bos="$bos" -v ctc="$ctc" '{
    $1 = "";
    CP[$0]++;
    CW[$2]++;
  }END{
    for (wp in CP) {
      n = split(wp, A, " ");
      w = A[1];

      printf("%-15s %f", w, CP[wp] / CW[w]);
      for (i = 2; i <= n; ++i) {
        printf(" %s", A[i]);
      }
      printf("\n");
    }
  }END{
    printf("%-15s %f %s\n", bos, 1.0, ctc);
  }' | sort;
  # ... and all characters from the out-of-vocabulary words.
  cut -d\  -f2- "$ctxt_out" | tr \  \\n | awk 'NF > 0' | sort -u |
  awk -v p="$char_prefix" '{
    printf("%-15s %f %s\n", p""$1, 1.0, $1);
  }';
) > "$4/lexiconp.txt";

# Add disambiguation symbols
tmp="$(mktemp)";
ndisambig=$("$SDIR/add_lex_disambig.pl" --pron-probs "$4/lexiconp.txt" "$tmp");
if [[ "$overwrite" = true || ! -s "$4/lexiconp_disambig.txt" ]] ||
     ! cmp -s "$tmp" "$4/lexiconp_disambig.txt"; then
  mv "$tmp" "$4/lexiconp_disambig.txt";
  overwrite=true;
fi;

# Character symbols
[[ "$overwrite" = false && -s "$4/chars.txt" ]] ||
sort -nk2 "$1" |
awk -v eps="$eps" -v nd="$ndisambig" 'BEGIN{
  N = 0;
  printf("%-5s %d\n", eps, N++);
}{
  printf("%-5s %d\n", $1, N++);
}END{
  for (d = 1; d <= nd; ++d) {
    printf("#%-4d %d\n", d, N++);
  }
}' > "$4/chars.txt";

# Word symbols
[[ "$overwrite" = false && -s "$4/words.txt" ]] ||
cut -d\  -f1 "$4/lexiconp.txt" |
awk -v bos="$bos" -v eos="$eos" -v eps="$eps" -v boov="$boov" -v eoov="$eoov" 'BEGIN{
  N = 0;
  printf("%-15s %d\n", eps, N++);
  VOC[eps] = 1;
}$1 != bos && $1 != eos && $1 != eps && !($1 in VOC){
  printf("%-15s %d\n", $1, N++);
}END{
  printf("%-15s %d\n", bos, N++);
  printf("%-15s %d\n", eos, N++);
  printf("%-15s %d\n", boov, N++);
  printf("%-15s %d\n", eoov, N++);
  printf("%-15s %d\n", "#0", N++);
}' > "$4/words.txt";

# Create integer list of disambiguation symbols.
gawk '$1 ~ /^#.+/{ print $2 }' "$4/chars.txt" > "$4/chars_disambig.int";
# Create integer list of disambiguation symbols.
gawk '$1 ~ /^#.+/{ print $2 }' "$4/words.txt" > "$4/words_disambig.int";

# Create HMM model and tree
"$SDIR/create_ctc_hmm_model.sh" \
  --eps "$eps" --ctc "$ctc" --overwrite "$overwrite" \
  "$4/chars.txt" "$4/model" "$4/tree";

# Create the lexicon FST with disambiguation symbols.
[[ "$overwrite" = false && -s "$4/L.fst" ]] ||
"$SDIR/make_lexicon_fst.pl" --pron-probs "$4/lexiconp_disambig.txt" |
fstcompile --isymbols="$4/chars.txt" --osymbols="$4/words.txt" |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstarcsort --sort_type=ilabel > "$4/L.fst" ||
{ echo "Failed $4/L.fst creation!" >&2 && exit 1; }

# Compose the context-dependent and the L transducers.
[[ "$overwrite" = false && -s "$4/CL.fst" ]] ||
fstcomposecontext --context-size=1 --central-position=0 \
		  --read-disambig-syms="$4/chars_disambig.int" \
		  --write-disambig-syms="$4/ilabels_disambig.int" \
		  "$4/ilabels" "$4/L.fst" |
fstarcsort --sort_type=ilabel > "$4/CL.fst" ||
{ echo "Failed $4/CL.fst creation!" >&2 && exit 1; }

# Create Ha transducer
[[ "$overwrite" = false && -s "$4/Ha.fst" ]] ||
make-h-transducer \
    --disambig-syms-out="$4/tid_disambig.int" \
    --transition-scale="$transition_scale" \
    "$4/ilabels" "$4/tree" "$4/model" > "$4/Ha.fst" ||
{ echo "Failed $4/Ha.fst creation!" >&2 && exit 1; }

# Create HaCL transducer.
[[ "$overwrite" = false && -s "$4/HCL.fst" ]] ||
fsttablecompose "$4/Ha.fst" "$4/CL.fst" |
fstdeterminizestar --use-log=true |
fstrmsymbols "$4/tid_disambig.int" |
fstrmepslocal |
fstminimizeencoded > "$4/HaCL.fst" ||
{ echo "Failed $4/HaCL.fst creation!" >&2 && exit 1; }

# Create HCL transducer.
[[ "$overwrite" = false && -s "$4/HCL.fst" ]] ||
add-self-loops --self-loop-scale="$loop_scale" --reorder=true \
	       "$4/model" "$4/HaCL.fst" |
fstarcsort --sort_type=olabel > "$4/HCL.fst" ||
{ echo "Failed $4/HCL.fst creation!" >&2 && exit 1; }

# Create Gw transducer from words (inc. oov token).
[[ "$overwrite" = false && -s "$4/Gw.fst" ]] ||
cut -d\  -f2 "$3" |
awk -v lex="$4/lexiconp.txt" -v eps="$eps" -v bos="$bos" -v eos="$eos" \
    -v oov="$boov" \
'BEGIN{
  while ((getline < lex) > 0) {
    LEX[$1] = 1;
  }
  if (bos in LEX) { has_bos = 1; }
  if (eos in LEX) { has_eos = 1; }
}{
  if (!($1 in LEX)) $1 = oov;
  CW[$1]++;
  CT++;
}END{
  print 0, 1, (has_bos ? bos : eps), eps;
  for (w in CW) {
    print 1, 2, w, w, log(CT) - log(CW[w]);
  }
  print 2, 3, (has_eos ? eos : eps), eps;
  print 3;
}' |
fstcompile --isymbols="$4/words.txt" --osymbols="$4/words.txt" |
fstrmepslocal --use-log=true |
fstarcsort --sort_type=ilabel > "$4/Gw.fst";

# Create Gc transducer from out-vocabulary words (inc. oov token).
bos_int="$(grep -w "$bos" "$4/words.txt" | cut -d\  -f2)";
eos_int="$(grep -w "$eos" "$4/words.txt" | cut -d\  -f2)";
rmsyms="$(echo "$bos_int $eos_int" | tr \  \\n)";
[[ "$overwrite" = false && -s "$4/Gc.fst" ]] ||
cut -d\  -f2-  "$ctxt_out" |
awk -v p="$char_prefix" '{ for (i=1; i<=NF; ++i) {$i=p""$i; } print; }' |
ngram-count $srilm_opts -text - -lm - |
arpa2fst --bos-symbol="$bos" --eos-symbol="$eos" --keep-symbols=false \
	 --disambig-symbol="#0" --read-symbol-table="$4/words.txt" /dev/stdin |
fstrmsymbols <(echo "$rmsyms") |
fstdeterminizestar --use-log=true |
fstminimizeencoded |
fstpushspecial |
fstrmsymbols <(gawk '$1 == "#0"{ print $2 }' "$4/words.txt") |
fstrmepslocal --use-log=true |
fstarcsort --sort_type=ilabel > "$4/Gc.fst";

# Replace OOV ark in Gw, with the Gc.
# Note1: We can do this explicitly because there is only one OOV arc in Gw.
# If Gw was a word n-gram, we could not do this statically.
# Note2: We do not use Gc directly, but only accept paths with at least 1 char.
boov_int="$(awk -v oov="$boov" '$1 == oov{print $2}' "$4/words.txt")";
eoov_int="$(awk -v oov="$eoov" '$1 == oov{print $2}' "$4/words.txt")";
[[ "$overwrite" = false && -s "$4/G_p$oov_penalty.fst" ]] ||
fstreplace --call_arc_labeling=output --return_arc_labeling=output \
  	   --return_label="$eoov_int" \
	   <(fstprint "$4/Gw.fst" | \
	     awk -v boov="$boov_int" -v p="$oov_penalty" '{
               if (NF >= 4 && $3 == boov) { $5 = (NF == 4 ? p : $5 + p); }
               print;
             }' | fstcompile) -1 \
	   <(fstcompose \
	       <(awk -v ex="^$char_prefix" '$1 ~ ex{
                   print 0, 1, $1;
                   print 1, 1, $1;
                 }END{
                   print 1;
                 }' "$4/words.txt" |
		 fstcompile --acceptor --isymbols="$4/words.txt" |
		 fstarcsort --sort_type=olabel) "$4/Gc.fst") "$boov_int" \
| fstarcsort --sort_type=ilabel > "$4/G_p$oov_penalty.fst";
