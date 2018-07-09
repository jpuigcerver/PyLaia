#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/src" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/src/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/src/parse_options.inc.sh file!" >&2 && exit 1;

batch_size=10;
gpu=1;
checkpoint="experiment.ckpt.lowest-valid-cer*";
fixed_height=true;
help_message="
Usage: ${0##*/} [options]

Options:
  --batch_size   : (type = integer, default = $batch_size)
                   Batch size for decoding.
  --gpu          : (type = integer, default = $gpu)
                   Select which GPU to use, index starts from 1.
                   Set to 0 for CPU.
  --checkpoint   : (type = str, default = $checkpoint)
                   Suffix of the checkpoint to use, can be a glob pattern.
  --fixed_height : (type = boolean, default = $fixed_height)
                   Use a fixed height model.
";
source "$(pwd)/src/parse_options.inc.sh" || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

if [ $gpu -gt 0 ]; then
  export CUDA_VISIBLE_DEVICES=$((gpu-1));
  gpu=1;
fi;

for f in data/lang/lines/{char,word}/aachen/{te,va}.txt train/{syms.txt,model}; do
  [ ! -s "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;

hasComputeWer=0;
if which compute-wer-bootci &> /dev/null; then hasComputeWer=1; fi;
[ $hasComputeWer -ne 0 ] ||
echo "WARNING: compute-wer-bootci was not found, so CER/WER won't be computed!" >&2;

mkdir -p decode/{forms,lines}/{char,word}/aachen;

for p in va te; do
  lines_char="decode/lines/char/aachen/${p}.txt";
  lines_word="decode/lines/word/aachen/${p}.txt";
  forms_char="decode/forms/char/aachen/${p}.txt";
  forms_word="decode/forms/word/aachen/${p}.txt";
  imgs_list="data/lists/lines/aachen/${p}.lst";

  # Decode lines
  pylaia-htr-decode-ctc \
    train/syms.txt \
    data/imgs/lines_h128 \
    "$imgs_list" \
    --train_path train \
    --join_str=" " \
    --gpu $gpu \
    --batch_size $batch_size \
    --checkpoint $checkpoint \
    --use_letters | sort -V > "$lines_char";
  # Note: The decoding step does not return the output
  # In the same order as the input unless batch size 1
  # is used. Sort must be done afterwards

  # Get word-level transcript hypotheses for lines
  gawk '{
    printf("%s ", $1);
    for (i=2;i<=NF;++i) {
      if ($i == "<space>")
        printf(" ");
      else
        printf("%s", $i);
    }
    printf("\n");
  }' "$lines_char" > "$lines_word";

  # Get form char-level transcript hypothesis
  gawk '{
    if (match($1, /^([^ ]+)-[0-9]+$/, A)) {
      if (A[1] != form_id) {
        if (form_id != "") printf("\n");
        form_id = A[1];
        $1 = A[1];
        printf("%s", $1);
      } else {
        printf(" %s", "<space>");
      }
      for (i=2; i<= NF; ++i) { printf(" %s", $i); }
    }
  }' < "$lines_char" > "$forms_char";

  # Get form word-level transcript hypothesis
  gawk '{
    if (match($1, /^([^ ]+)-[0-9]+$/, A)) {
      if (A[1] != form_id) {
        if (form_id != "") printf("\n");
        form_id = A[1];
        $1 = A[1];
        printf("%s", $1);
      }
      for (i=2; i<= NF; ++i) { printf(" %s", $i); }
    }
  }' < "$lines_word" > "$forms_word";
done;

if [ $hasComputeWer -eq 1 ]; then
  rm -f decode/decode.out;
  for i in lines forms; do
    for j in char word; do
      for k in va te; do
        # Compute CER and WER using Kaldi's compute-wer-bootci
        compute-wer-bootci --print-args=false\
          "ark:data/lang/${i}/${j}/aachen/${k}.txt" \
          "ark:decode/${i}/${j}/aachen/${k}.txt" | \
        awk -v i=$i -v j=$j -v k=$k '{ $1=""; $2=":"; print i"/"j"/"k$0}' | \
        tee -a decode/decode.out;
      done;
    done;
  done;
fi;
