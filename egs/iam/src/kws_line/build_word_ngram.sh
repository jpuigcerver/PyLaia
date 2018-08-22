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
ngram_order=3;
overwrite=false;
voc_size=50000;
voc_tune=false;
help_message="
Usage: ${0##*/} [options] syms_ctc output_dir
  --lowercase     : (type = boolean, default = $lowercase)
                    If true, train a model of lowercase only words.
  --ngram_method  : (type = string, default = \"$ngram_method\")
                    Method used to build the n-gram model (see ngrammake).
  --ngram_order   : (type = integer, default = $ngram_order)
                    Order of the n-gram language model.
  --overwrite     : (type = boolean, default = $overwrite)
                    Overwrite previously created files.
  --voc_size      : (type = integer, default = $voc_size)
                    Keep this number of most frequent words.
  --voc_tune      : (type = boolean, default = $voc_tune)
                    If true, use hyperopt to tune the vocabulary.
";
source ../utils/parse_options.inc.sh || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

syms_ctc="$1";
output_dir="$2";

if [ "$lowercase" = true ]; then
  syms_ctc=data/kws_line/lang/char/syms_ctc_lowercase.txt;
  te_txt=data/kws_line/lang/word/te_lowercase.txt;
  tr_txt=data/kws_line/lang/word/tr_lowercase.txt;
  va_txt=data/kws_line/lang/word/va_lowercase.txt;
  external_txt=( \
    data/kws_line/lang/external/word/brown_lowercase.txt \
    data/kws_line/lang/external/word/lob_excludealltestsets_lowercase.txt \
    data/kws_line/lang/external/word/wellington_lowercase.txt \
  );
else
  syms_ctc=data/kws_line/lang/char/syms_ctc.txt;
  te_txt=data/kws_line/lang/word/te.txt;
  tr_txt=data/kws_line/lang/word/tr.txt;
  va_txt=data/kws_line/lang/word/va.txt;
  external_txt=( \
    data/kws_line/lang/external/word/brown.txt \
    data/kws_line/lang/external/word/lob_excludealltestsets.txt \
    data/kws_line/lang/external/word/wellington.txt \
  );
fi;

check_all_files -s "$syms_ctc" "$te_txt" "$tr_txt" "$va_txt" \
                   "${external_txt[@]}" || exit 1;
mkdir -p "$output_dir" || exit 1;

function filter_words_missing_symbol () {
  tr \  \\n | gawk -v syms_ctc="$1" 'BEGIN{
    while((getline < syms_ctc) > 0) {
      CHAR[$1] = 1;
    }
  }{
    ok=1;
    for (i=1; i <= length($0) && ok; ++i) {
      if (!(substr($0, i, 1) in CHAR)) { ok = 0; }
    }
    if (ok) print;
  }'
  return 0;
}

# Create list of words
[ "$overwrite" = false -a -s "$output_dir/words.txt" ] || {
  if [ "$voc_tune" = true ]; then
    tmp="$(mktemp)";
    cut -d\  -f2- "$tr_txt" | filter_words_missing_symbol "$syms_ctc" > "$tmp";
    external_txt2=("$tmp");
    for f in "${external_txt[@]}"; do
      tmp="$(mktemp)";
      cat "$f" | filter_words_missing_symbol "$syms_ctc" > "$tmp";
      external_txt2+=("$tmp");
    done;
    ./src/kws_line/create_vocabulary.py \
      "$voc_size" "${external_txt2[@]}" <(cut -d\  -f2- "$va_txt");
    rm -f "${external_txt2[@]}";
  else
    { cut -d\  -f2- "$tr_txt"; cat "${external_txt[@]}"; } |
    tr \  \\n |
    filter_words_missing_symbol "$syms_ctc" |
    sort | uniq -c | sort -nrk1 | head -n "$voc_size" | awk '{print $2}';
  fi |
  sort |
  gawk 'BEGIN{
    N = 0;
    print "<eps>", N++;
    print "<unk>", N++;
    print "<s>", N++;
    print "</s>", N++;
  }{
    print $1, N++;
  }END{
    print "#0", N++;
  }' > "$output_dir/words.txt";
} || exit 1;

# Create word lexicon
[ "$overwrite" = false -a -s "$output_dir/lexicon.txt" ] ||
gawk 'BEGIN{
  IGNORE["<eps>"] = 1;
  IGNORE["<unk>"] = 1;
  IGNORE["<s>"] = 1;
  IGNORE["</s>"] = 1;
  IGNORE["#0"] = 1;
}!($1 in IGNORE){
  printf("%20-s <space>", $1);
  for (j = 1; j <= length($1); ++j) {
    printf(" %s", substr($1, j, 1));
  }
  printf("\n");
}' "$output_dir/words.txt" > "$output_dir/lexicon.txt" || exit 1;


# Create char symbols, including disambiguation symbols
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
    for (d = 0; d <= nd; ++d) {
      printf("#%d %d\n", d, NR + d + 1);
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
sym2int.pl -f 1-2 "$output_dir/words.txt" |
sym2int.pl -f 3- "$output_dir/chars.txt" > "$output_dir/lexicon_align.txt" || exit 1;

# Count n-grams of training data
c="$(basename "$tr_txt" .txt)";
[ "$overwrite" = false -a -s "$output_dir/$c.count.fst" ] || {
  tmp="$(mktemp)";
  cut -d\  -f2- "$tr_txt" > "$tmp";
  farcompilestrings \
    --symbols="$output_dir/words.txt" --unknown_symbol="<unk>" --keep_symbols=true \
    "$tmp" |
  ngramcount --order="$ngram_order" > "$output_dir/$c.count.fst";
  rm "$tmp";
} || exit 1;

# Count n-grams of external data
for f in "${external_txt[@]}"; do
  c="$(basename "$f" .txt)";
  [ "$overwrite" = false -a -s "$output_dir/$c.count.fst" ] || {


    farcompilestrings \
      --symbols="$output_dir/words.txt" --unknown_symbol="<unk>" --keep_symbols=true \
      "$f" |
    ngramcount --order="$ngram_order" > "$output_dir/$c.count.fst"
  } || exit 1;
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
    --symbols="$output_dir/words.txt" --unknown_symbol="<unk>" \
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
