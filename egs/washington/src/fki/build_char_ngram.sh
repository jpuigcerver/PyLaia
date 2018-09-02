#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/../..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";

check_opengrm || exit 1;

ctc="<ctc>";
eps="<eps>";
ngram_method=kneser_ney;
ngram_order=6;
overwrite=false;
unk="<unk>";
help_message="
Usage: ${0##*/} [options] syms_ctc tr_txt va_txt te_txt output_dir

Options:
  --ctc           : (type = string, default = \"$ctc\")
                    Symbol representing the CTC label.
  --eps           : (type = string, default = \"$eps\")
                    Symbol representing the epsilon label.
  --ngram_method  : (type = string, default = \"$ngram_method\")
                    Method used to build the n-gram model (see ngrammake).
  --ngram_order   : (type = integer, default = $ngram_order)
                    Order of the n-gram language model.
  --overwrite     : (type = boolean, default = $overwrite)
                    Overwrite previously created files.
  --unk           : (type = string, default = \"$unk\")
                    Symbol representing unknown words.
";
source $PWD/../utils/parse_options.inc.sh || exit 1;
[ $# -ne 5 ] && echo "$help_message" >&2 && exit 1;

syms_ctc="$1";
tr_txt="$2";
va_txt="$3";
te_txt="$4";
output_dir="$5";

check_all_files -s "$syms_ctc" "$te_txt" "$tr_txt" "$va_txt" || exit 1;
mkdir -p "$output_dir" || exit 1;

# Create the lexicon file
[ "$overwrite" = false -a -s "$output_dir/lexicon.txt" ] ||
gawk -v ctc="$ctc" -v eps="$eps" '{
  if ($1 != "$eps" && $1 != "$ctc") {
    printf("%8-s %s\n", $1, $1);
  }
}' "$syms_ctc" > "$output_dir/lexicon.txt";

# Create char symbols, including disambiguation symbols and <unk>
[ "$overwrite" = false -a \
  -s "$output_dir/lexicon_disambig.txt" -a \
  -s "$output_dir/chars.txt" ] || {
  ndisambig=$(add_lex_disambig.pl "$output_dir/lexicon.txt" \
              "$output_dir/lexicon_disambig.txt");
  gawk -v eps="$eps" -v nd="$ndisambig" -v unk="<unk>" 'BEGIN{
    print eps, 0;
  }{
    print $1, NR;
  }END{
    print unk, NR + 1;
    for (d = 0; d <= nd; ++d) {
      printf("#%d %d\n", d, NR + d + 2);
    }
  }' "$syms_ctc" > "$output_dir/chars.txt";
}

# Create alignment lexicon (to use with lattice-align-words-lexicon)
[ "$overwrite" = false -a -s "$output_dir/lexicon_align.txt" ] ||
awk '{
  printf("%10-s %10-s", $1, $1);
  for (i = 2; i <= NF; ++i) { printf(" %s", $i); }
  printf("\n");
}' "$output_dir/lexicon.txt" |
sym2int.pl -f 1- "$output_dir/chars.txt" > "$output_dir/lexicon_align.txt" ||
exit 1;

# Make Train-only LM fst, ignoring <unk> token
[ "$overwrite" = false -a -s "$output_dir/lm.fst" ] || {
  tmp="$(mktemp)";
  cut -d\  -f2- "$tr_txt" > "$tmp";
  farcompilestrings \
    --symbols="$output_dir/chars.txt" \
    --unknown_symbol="$unk" \
    --keep_symbols=true \
    "$tmp" |
  ngramcount --order="$ngram_order" |
  ngramprint --integers | grep -v "$unk" |
  ngramread --symbols="$output_dir/chars.txt" --OOV_symbol="$unk" \
            --epsilon_symbol="$eps" |
  ngrammake --method="$ngram_method" > "$output_dir/lm.fst";
  rm "$tmp";
} || exit 1;

function compute_ppl () {
  tmp="$(mktemp)";
  cut -d\  -f2- "$1" > "$tmp";
  farcompilestrings \
    --symbols="$output_dir/chars.txt" --unknown_symbol="<unk>" \
    --keep_symbols=true "$tmp" |
  ngramperplexity "$2" 2> /dev/null |
  gawk -v txt="$1" '{
    if (match($0, /, ([0-9]+) words, ([0-9]+) OOVs/, a)) {
       words = a[1];
       oov = a[2];
    } else if (match($0, /perplexity = ([0-9.]+)/, a)) {
       ppl = a[1];
    }
  }END{
    poov = oov / words * 100;
    printf("data = %s ; ppl = %g ; oov = %d (%.2f%)\n", txt, ppl, oov, poov);
  }' || return 1;
  rm "$tmp";
  return 0;
}

# Compute validation and test perplexity and running OOV
for txt in "$va_txt" "$te_txt"; do
  compute_ppl "$txt" "$output_dir/lm.fst" || exit 1;
done;
