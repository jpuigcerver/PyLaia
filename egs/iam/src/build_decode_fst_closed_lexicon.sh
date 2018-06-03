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
use_priors=true;
help_message="
Usage: ${0##*/} [options] <syms> <char_gt> <word_gt> <output_dir>

Arguments:
  syms         : Table of symbols used for the CTC training.
  char_gt      :
  word_gt      :
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
  --use_priors       : (type = boolean, value = $use_priors)
                       If true, add the word priors to G.
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -ne 4 ] && echo "$help_message" >&2 && exit 1;

for f in "$1" "$2" "$3"; do
  [ ! -s "$f" ] &&
  echo "ERROR: File \"$f\" does not exist or is empty!" >&2 && exit 1;
done
mkdir -p "$4";

[[ "$overwrite" = false && -s "$4/lexiconp.txt" ]] ||
join -1 1 <(sort "$3") <(sort "$2") |
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
}' | sort -V > "$4/lexiconp.txt";

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
cut -d\  -f1 "$4/lexiconp.txt" | sort -uV |
awk -v bos="$bos" -v eos="$eos" -v eps="$eps" 'BEGIN{
  N = 0;
  printf("%-15s %d\n", eps, N++);
}$1 != bos && $1 != eos && $1 != eps{
  printf("%-15s %d\n", $1, N++);
}END{
  printf("%-15s %d\n", bos, N++);
  printf("%-15s %d\n", eos, N++);
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

# Create G transducer.
[[ "$overwrite" = false && -s "$4/G.fst" ]] ||
cut -d\  -f2 "$3" |
awk -v eps="$eps" -v bos="$bos" -v eos="$eos" -v ap="$use_priors" '{
  CW[$1]++;
  CT++;
}END{
  print 0, 1, bos, eps;
  for (w in CW) {
    if (ap == "true") {
      print 1, 2, w, w, log(CT) - log(CW[w]);
    } else {
      print 1, 2, w, w;
    }
  }
  print 2, 3, eos, eps;
  print 3;
}' |
fstcompile --isymbols="$4/words.txt" --osymbols="$4/words.txt" |
fstrmsymbols <(cut -d\  -f1 "$4/lexiconp.txt" |
	       awk -v bos="$bos" -v eos="$eos" 'BEGIN{
                 has_bos = has_eos = 0;
               }{
                 if ($1 == bos) has_bos=1;
                 if ($1 == eos) has_eos=1;
               }END{
                 if (!has_bos) print bos;
                 if (!has_eos) print eos;
               }' | "$SDIR/sym2int.pl" -f 1 "$4/words.txt") |
fstrmepslocal --use-log=true |
fstarcsort --sort_type=ilabel > "$4/G.fst";
