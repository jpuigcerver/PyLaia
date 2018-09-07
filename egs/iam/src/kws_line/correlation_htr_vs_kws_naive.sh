#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

help_message="
Usage: ${0##*/}
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

queries=data/kws_line/lang/queries/iam_old_papers_queries.txt;
check_all_files -s "$queries" \
                   data/kws_line/lang/delimiters.txt \
                   data/kws_line/lang/kws_refs/va.txt \
                   data/kws_line/lang/kws_refs/te.txt \
                   data/kws_line/lang/char/va.txt \
                   data/kws_line/lang/char/te.txt || exit 1;

tmp="$(mktemp)";
for n in 3 4 5 6 7 8; do
  check_all_files -s "data/kws_line/lm/char_${n}gram/chars.txt" \
                     "data/kws_line/lm/char_${n}gram/model" || exit 1;
  readarray -t delimiters < \
      <(join -1 1 <(sort -k1b,1 data/kws_line/lang/delimiters.txt) \
                  <(sort -k1b,1 "data/kws_line/lm/char_${n}gram/chars.txt") |
        sort -nk2 | awk '{print $2}') || exit 1;
  for ps in $(seq 0.0 0.2 1.0); do
    latdir=data/kws_line/lats/resize_h128/char_${n}gram/ps${ps};
    check_all_dirs "$latdir" || exit 1;
    check_all_files -s "$latdir/va.lat.ark" "$latdir/te.lat.ark" || exit 1;
    for as in 0.6 0.8 1.0 1.2; do
      for p in va te; do
        # CER/WER
        ./src/kws_line/compute_metric_htr.py \
          --acoustic-scale="$as" \
          --wspace="<space>" \
          --char-separator="" \
          "data/kws_line/lm/char_${n}gram/chars.txt" \
          "data/kws_line/lm/char_${n}gram/model" \
          "$latdir/${p}.lat.ark" \
          "data/kws_line/lang/char/${p}.txt";
        # Naive indexing from the 1-best recognition
        lattice-to-phone-lattice \
          --print-args=false \
          "data/kws_line/lm/char_${n}gram/model" \
          "ark:$latdir/${p}.lat.ark" ark:- 2> /dev/null |
        lattice-best-path \
          --print-args=false \
          --acoustic-scale="$as" \
          ark:- ark,t:- 2>/dev/null |
        int2sym.pl -f 2- "data/kws_line/lm/char_${n}gram/chars.txt" |
        sed -r "s| <space>( <space>)+| <space>|g" |
        awk -v ws="<space>" \
          '{ if($2 == ws) {$2="";} if($NF == ws) {$NF="";} print; }' |
        awk -v ws="<space>" -v cs="" '{
          printf("%s ", $1);
          for (i = 2; i <= NF; ++i) {
            if ($i == ws) { printf(" "); }
            else { printf("%s%s", cs, $i); }
          }
          printf("\n");
        }' |
        awk -v reffile="data/kws_line/lang/kws_refs/$p.txt" '
        BEGIN{
          while((getline < reffile) > 0) {
            REF[$2" "$1] = 1;
          }
        }{
          for (i = 2; i <= NF; ++i) {
            pair = $1" "$i;
            if (pair in REF) {
              print $1, $i, 1, 1.0;
              HYP[pair] = 1;
            } else {
              print $1, $i, 0, 1.0;
            }
          }
        }END{
          for (pair in REF) {
            if (!(pair in HYP)) {
              print pair, 1, "-inf";
            }
          }
        }' | sort -u  > "$tmp";
        kws-assessment-joan -a -m -t -w "$queries" "$tmp" 2> /dev/null |
        awk '{
          if(match($0, /(AP|MAP) = ([0-9.]+)/, aa)) {
            R[aa[1]] = aa[2];
          }
        }END{
          printf("'\''mAP'\'': %f  '\''gAP'\'': %f\n",
                 R["MAP"], R["AP"]);
        }' || exit 1;
      done |
      gawk -v n="$n" -v as="$as" -v ps="$ps" '
      BEGIN{
        mAP[0] = mAP[1] = gAP[0] = gAP[1] = -1;
        CER[0] = CER[1] = WER[0] = WER[1] = -1;
        Nap = Ner = 0;
      }{
        if (match($0, "'\''mAP'\'': ([0-9.]+)", aa) &&
            match($0, "'\''gAP'\'': ([0-9.]+)", bb)) {
          mAP[Nap] = aa[1];
          gAP[Nap] = bb[1];
          Nap++;
        } else if (match($0, "'\''CER'\'': ([0-9.]+)", aa) &&
                   match($0, "'\''WER'\'': ([0-9.]+)", bb)) {
          CER[Ner] = aa[1];
          WER[Ner] = bb[1];
          Ner++;
        } else {
          print "Wrong last line: "$0 > "/dev/stderr"; exit(1);
        }
      }END{
        if (Nap != 2 || Ner != 2) {
          print "Missing results!" > "/dev/stderr"; exit(1);
        }
        printf("%2-d %.1f %.1f %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f\n",
               n, as, ps,
               mAP[0] * 100, gAP[0] * 100, mAP[1] * 100, gAP[1] * 100,
               CER[0], WER[0], CER[1], WER[1]);
      }'
    done;
  done;
done;
