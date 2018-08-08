#!/usr/bin/env bash
set -e;
export LC_NUMERIC=C;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";

source "../utils/functions_check.inc.sh" || exit 1;

overwrite=false;
wspace="<space>";
help_message="
Usage: ${0##*/} [options] text1 [text2 ...]
Arguments:
  text1 ...    : External text files to process
                 (e.g. data/external/brown.txt data/external/wellington.txt).
Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
  --wspace     : (type = string, default \"$wspace\")
                 Use this symbol to represent the whitespace character.
";
source "../utils/parse_options.inc.sh" || exit 1;
[ $# -lt 1 ] && echo "$help_message" >&2 && exit 1;

check_all_files -s "${@}" || exit 1;

mkdir -p data/kws_line/lang/external/char data/kws_line/lang/external/word;

# 1. We convert some special sequences to UTF-8 characters, such as a*?1 -> ä,
# a*?2 -> á, a*?3 -> à, n*?4 -> ñ, etc.
# 2. Since IAM does not contain these characters, transliterate all UTF-8 codes
# to reduce the number of tokens in the LM.
# 3. Put abbrev. like 's, 't, 'd, etc. together with their word,.
# 4. Finally, convert the original word-level transcript (word/external/$c.txt)
# to a character-level transcript (char/external/$c.txt).
for f in "${@}"; do
  c="$(basename "$f")"; c="${c%.*}";
  wtxt="data/kws_line/lang/external/word/${c}.txt";

  [ "$overwrite" = false -a -s "$wtxt" ] ||
  cat "$f" |
  sed -r 's|a\*\?1|ä|g;' |
  sed -r 's|a\*\?2|á|g;s|e\*\?2|é|g;s|i\*\?2|í|g;s|o\*\?2|ó|g;s|u\*\?2|ú|g;s|A\*\?2|Á|g;s|E\*\?2|É|g;s|I\*\?2|Í|g;s|O\*\?2|Ó|g;s|U\*\?2|Ú|g;' |
  sed -r 's|a\*\?3|à|g;s|e\*\?3|è|g;s|i\*\?3|ì|g;s|o\*\?3|ò|g;s|u\*\?3|ù|g;s|A\*\?3|À|g;s|E\*\?3|È|g;s|I\*\?3|Ì|g;s|O\*\?3|Ò|g;s|U\*\?3|Ù|g;' |
  sed -r 's|a\*\?4|ã|g;s|A\*\?4|Ã|g;s|n\*\?4|ñ|g;s|N\*\?4|Ñ|g;' |
  sed -r 's|a\*\?5|â|g;s|e\*\?5|ê|g;s|i\*\?5|î|g;s|o\*\?5|ô|g;s|u\*\?5|û|g;s|A\*\?5|Â|g;s|E\*\?5|Ê|g;s|I\*\?5|Î|g;s|O\*\?5|Ô|g;s|U\*\?5|Û|g;' |
  sed -r 's|c\*\?6|ç|g;s|C\*\?6|Ç|g;' |
  sed -r 's|s\*\?10|š|g;s|S\*\?10|Š|g;' |
  sed -r 's|l\*\?11|ł|g;s|L\*\?11|Ł|g;' |
  iconv -f utf-8 -t ascii//TRANSLIT |
  sed 's/ '\''\(s\|d\|ll\|m\|ve\|t\|re\|S\|D\|LL\|M\|VE\|T\|RE\)\b/'\''\1/g' \
  > "$wtxt" ||
  { echo "ERROR: Creating file \"$wtxt\"!" >&2 && exit 1; }

  ctxt="data/kws_line/lang/external/char/${c}.txt";
  [ "$overwrite" = false -a -s "$ctxt" ] ||
  gawk -v ws="$wspace" '{
    for(i=1;i<=NF;++i) {
      for(j=1;j<=length($i);++j) {
        printf(" %s", substr($i, j, 1));
      }
      if (i < NF) printf(" %s", ws);
    }
    printf("\n");
  }' "$wtxt" > "$ctxt" ||
  { echo "ERROR: Creating file \"$ctxt\"!" >&2 && exit 1; }

  wtxt_lc="data/kws_line/lang/external/word/${c}_lowercase.txt";
  [ "$overwrite" = false -a -s "$wtxt_lc" ] ||
  gawk '{ print tolower($0); }' $wtxt > "$wtxt_lc" ||
  { echo "ERROR: Creating file \"$wtxt_lc\"!" >&2 && exit 1; }

  ctxt_lc="data/kws_line/lang/external/char/${c}_lowercase.txt";
  [ "$overwrite" = false -a -s "$ctxt_lc" ] ||
  gawk '{ print tolower($0); }' $ctxt > "$ctxt_lc" ||
  { echo "ERROR: Creating file \"$ctxt_lc\"!" >&2 && exit 1; }
done;
