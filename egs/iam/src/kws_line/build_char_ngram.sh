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

lowercase=false;
ngram_method=kneser_ney;
ngram_order=6;
overwrite=false;
help_message="
Usage: ${0##*/} [options] output_dir
  --lowercase     : (type = boolean, default = $lowercase)
                    If true, train a model of lowercase only characters.
  --ngram_method  : (type = string, default = \"$ngram_method\")
                    Method used to build the n-gram model (see ngrammake).
  --ngram_order   : (type = integer, default = $ngram_order)
                    Order of the n-gram language model.
  --overwrite     : (type = boolean, default = $overwrite)
                    Overwrite previously created files.
";
source ../utils/parse_options.inc.sh || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

output_dir="$1";

if [ "$lowercase" = true ]; then
  syms_ctc=data/kws_line/lang/char/syms_ctc_lowercase.txt;
  te_txt=data/kws_line/lang/char/te_lowercase.txt;
  tr_txt=data/kws_line/lang/char/tr_lowercase.txt;
  va_txt=data/kws_line/lang/char/va_lowercase.txt;
  external_txt=( \
    data/kws_line/lang/external/char/brown_lowercase.txt \
    data/kws_line/lang/external/char/lob_excludealltestsets_lowercase.txt \
    data/kws_line/lang/external/char/wellington_lowercase.txt \
  );
else
  syms_ctc=data/kws_line/lang/char/syms_ctc.txt;
  te_txt=data/kws_line/lang/char/te.txt;
  tr_txt=data/kws_line/lang/char/tr.txt;
  va_txt=data/kws_line/lang/char/va.txt;
  external_txt=( \
    data/kws_line/lang/external/char/brown.txt \
    data/kws_line/lang/external/char/lob_excludealltestsets.txt \
    data/kws_line/lang/external/char/wellington.txt \
  );
fi;

check_all_files -s "$syms_ctc" "$te_txt" "$tr_txt" "$va_txt" \
                   "${external_txt[@]}" || exit 1;
mkdir -p "$output_dir" || exit 1;

# Create the lexicon file
[ "$overwrite" = false -a -s "$output_dir/lexicon.txt" ] ||
gawk '{
  if ($1 != "<eps>" && $1 != "<ctc>") {
    printf("%8-s %s\n", $1, $1);
  }
}' "$syms_ctc" > "$output_dir/lexicon.txt";

# Create char symbols, including disambiguation symbols and <unk>
[ "$overwrite" = false -a \
  -s "$output_dir/lexicon_disambig.txt" -a \
  -s "$output_dir/chars.txt" ] || {
  ndisambig=$(add_lex_disambig.pl "$output_dir/lexicon.txt" \
              "$output_dir/lexicon_disambig.txt");
  gawk -v nd="$ndisambig" 'BEGIN{
    print "<eps>", 0;
  }{
    print $1, NR;
  }END{
    print "<unk>", NR + 1;
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
sym2int.pl -f 1- "$output_dir/chars.txt" > "$output_dir/lexicon_align.txt" || exit 1;

# Count n-grams of training data
c="$(basename "$tr_txt" .txt)";
[ "$overwrite" = false -a -s "$output_dir/$c.count.fst" ] || {
  tmp="$(mktemp)";
  cut -d\  -f2- "$tr_txt" > "$tmp";
  farcompilestrings \
    --symbols="$output_dir/chars.txt" --unknown_symbol="<unk>" --keep_symbols=true \
    "$tmp" |
  ngramcount --order="$ngram_order" > "$output_dir/$c.count.fst";
  rm "$tmp";
} || exit 1;

# Count n-grams of external data
for f in "${external_txt[@]}"; do
  c="$(basename "$f" .txt)";
  [ "$overwrite" = false -a -s "$output_dir/$c.count.fst" ] ||
  farcompilestrings \
    --symbols="$output_dir/chars.txt" --unknown_symbol="<unk>" --keep_symbols=true \
    "$f" |
  ngramcount --order="$ngram_order" > "$output_dir/$c.count.fst" || exit 1;
done;

lm_files=();
info_files=();
for f in "$tr_txt" "${external_txt[@]}"; do
  c="$(basename "$f" .txt)";

  # Make individual ARPA files, ignoring <unk> token
  [ "$overwrite" = false -a -s "$output_dir/$c.lm.fst" ] ||
  ngramprint --integers "$output_dir/$c.count.fst" | grep -v "<unk>" | ngramread |
  ngrammake --method="$ngram_method" > "$output_dir/$c.lm.fst" || exit 1;

  # Remove n-grams containing <unk>
  [ "$overwrite" = false -a -s "$output_dir/$c.lm.info" ] ||
  ngramprint --ARPA "$output_dir/$c.lm.fst" |
  ngram -order "$ngram_order" -debug 2 -ppl <(cut -d\  -f2- "$va_txt") -lm - \
    &> "$output_dir/$c.lm.info" ||
  { echo "ERROR: Computing individual LM perplexity." >&2 && exit 1; }

  lm_files+=("$output_dir/$c.lm.fst");
  info_files+=("$output_dir/$c.lm.info");
done;

# Compute interpolation weights
[ "$overwrite" = false -a -s "$output_dir/best_mix.info" ] ||
compute-best-mix "${info_files[@]}" &> "$output_dir/best_mix.info" ||
{ echo "ERROR: Computing LM mixture weights." >&2 && exit 1; }
lambdas=( $(grep "best lambda" "$output_dir/best_mix.info" | \
            gawk -F\( '{print $2}' | tr -d \)) );

# Interpolate language models
[ "$overwrite" = false -a -s "$output_dir/lm.fst" ] ||
ngram -order "$ngram_order" \
      -lm <(ngramprint --ARPA "${lm_files[0]}") \
      -mix-lm <(ngramprint --ARPA "${lm_files[1]}") \
      -mix-lm2 <(ngramprint --ARPA "${lm_files[2]}") \
      -mix-lm3 <(ngramprint --ARPA "${lm_files[3]}") \
      -lambda "${lambdas[0]}" \
      -mix-lambda2 "${lambdas[2]}" \
      -mix-lambda3 "${lambdas[3]}" \
      -write-lm - |
ngramread --ARPA > "$output_dir/lm.fst" || exit 1;

# Compute validation and test perplexity and running OOV
for txt in "$va_txt" "$te_txt"; do
  tmp="$(mktemp)";
  cut -d\  -f2- "$txt" > "$tmp";
  farcompilestrings \
    --symbols="$output_dir/chars.txt" --unknown_symbol="<unk>" \
    --keep_symbols=true "$tmp" |
  ngramperplexity "$output_dir/lm.fst" 2> /dev/null |
  gawk -v txt="$txt" '{
    if (match($0, /, ([0-9]+) words, ([0-9]+) OOVs/, a)) {
       words = a[1];
       oov = a[2];
    } else if (match($0, /perplexity = ([0-9.]+)/, a)) {
       ppl = a[1];
    }
  }END{
    poov = oov / words * 100;
    printf("data = %s ; ppl = %g ; oov = %d (%.2f%)\n", txt, ppl, oov, poov);
  }' || exit 1;
  rm "$tmp";
done;
