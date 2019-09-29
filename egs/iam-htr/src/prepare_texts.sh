#!/bin/bash
set -e;

# Directory where the script is placed.
source "../utils/functions_check.inc.sh" || exit 1;

overwrite=false;
wspace="@";
help_message="
Usage: ${0##*/} [options]

Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
  --wspace     : (type = string, default = \"$wspace\")
                 Use this symbol to represent the whitespace character.
";
source "../utils/parse_options.inc.sh" || exit 1;

check_all_programs cut gawk sed sort tr || exit 1;

partitions=(lines sentences words);
cut_fields=(1,9- 1,10- 1,9-);
for p in $(seq ${#partitions[@]}); do
  partition="${partitions[p-1]}";
  mkdir -p "data/lang/all/$partition";
  all_ch="data/lang/all/$partition/char.txt";
  all_wo="data/lang/all/$partition/word.txt";

  ###################################################
  # Prepare word-level transcripts.
  ###################################################
  [[ "$overwrite" = false && -s "${all_wo}" &&
     ( ! "${all_wo}" -ot "data/original/$partition.txt" ) ]] ||
  gawk '$1 !~ /^#/' "data/original/$partition.txt" |
  cut -d\  -f"${cut_fields[p-1]}" |
  gawk '{ $1=$1"|"; print; }' |
  # Some words include spaces (e.g. "B B C" -> "BBC"), remove them.
  sed -r 's| +||g' |
  # Replace character | with whitespace in all line.
  tr \| \  |
  # Separate ID with |, again.
  sed 's/^\([^ ]\+\) \(.*\)$/\1|\2/g' |
  # Some contractions where separated from the words to reduce the vocabulary
  # size. These separations are unreal, we join them (e.g. "We 'll" -> "We'll").
  sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' |
  # Replace character | with whitespace in all line.
  tr \| \  |
  sort -k1 > "${all_wo}" ||
  { echo "ERROR: Creating file \"${all_wo}\"!" >&2; exit 1; }

  ###################################################
  # Prepare character-level transcripts.
  ###################################################
  [[ "$overwrite" = false && -s "${all_ch}" &&
     ( ! "${all_ch}" -ot "${all_wo}" ) ]] ||
  gawk -v ws="$wspace" '{
    printf("%s", $1);
    for(i=2;i<=NF;++i) {
      for(j=1;j<=length($i);++j) {
        printf(" %s", substr($i, j, 1));
      }
      if (i < NF) printf(" %s", ws);
    }
    printf("\n");
  }' "${all_wo}" |
  sort -k1 > "${all_ch}" ||
  { echo "ERROR: Creating file \"${all_ch}\"!" >&2; exit 1; }
done;


function join_lines () {
  [[ $# -ne 2 && $# -ne 3 ]] && \
  echo "Usage: ${0##*/} [sep] input output" >&2 && return 1;

  local sep=;
  [[ $# -eq 3 ]] && { sep="$1"; shift; }

  gawk -v sep="$sep" '
  BEGIN{
    form_id="";
  }
  {
    if (match($0, /^([^ ]+)-[0-9]+ (.+)$/, A)) {
      if (A[1] != form_id) {
        if (form_id != "") printf("\n");
        printf("%s %s", A[1], A[2]);
        form_id = A[1];
      } else {
        printf("%s %s", sep, A[2]);
      }
    } else {
      print "Unexpected line: "$0 > "/dev/stderr";
      exit(1);
    }
  }
  END{
    if (form_id != "") printf("\n");
  }' "$1" > "$2";
  return 0;
}


# Split IAM into different test/train/valid files.
for s in graves pham puigcerver; do
  for u in char word; do
    # Prepare line transcripts.
    odir="data/lang/$s/lines/$u";
    mkdir -p "$odir";
    for p in te tr va; do
      [[ "$overwrite" = false && -s "$odir/$p.txt" &&
	 ( ! "$odir/$p.txt" -ot "data/lang/all/lines/$u.txt" ) ]] ||
      join -1 1 "data/splits/$s/$p.lst" "data/lang/all/lines/$u.txt" \
	   > "$odir/$p.txt" ||
      { echo "ERROR: Creating file \"$odir/$p.txt\"!" >&2 && exit 1; }
    done;

    # Prepare form transcripts.
    sep=; [ "$u" = char ] && sep=" $wspace";
    odir="data/lang/$s/forms/$u";
    mkdir -p "$odir";
    for p in te tr va; do
      [[ "$overwrite" = false && -s "$odir/$p.txt" &&
	 ( ! "$odir/$p.txt" -ot "data/lang/$s/lines/$u/$p.txt" ) ]] ||
      join_lines "$sep" "data/lang/$s/lines/$u/$p.txt" "$odir/$p.txt" ||
      { echo "ERROR: Creating file \"$odir/$p.txt\"!" >&2 && exit 1; }
    done;
  done;
done;
