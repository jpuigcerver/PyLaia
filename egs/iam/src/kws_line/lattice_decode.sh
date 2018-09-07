#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";

acoustic_scale=1.0;
wspace="<space>";
char_sep="";
help_message="
Usage: ${0##*/} <fst_dir> <lattice_ark>
Options:
  --acoustic_scale : (type = float, default = $acoustic_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 2 ] && echo "$help_message" >&2 && exit 1;

check_all_files -s "$2" "$1/model" "$1/chars.txt" || exit 1;

lattice-to-phone-lattice --print-args=false "$1/model" ark:"$2" ark:- |
lattice-best-path --print-args=false --acoustic-scale="$acoustic_scale" \
  ark:- ark,t:- 2>/dev/null |
int2sym.pl -f 2- "$1/chars.txt" |
sed -r "s| $wspace( $wspace)+| $wspace|g" |
awk -v ws="$wspace" '{ if($2 == ws) $2=""; if($NF == ws) $2=""; print; }' |
awk -v ws="$wspace" -v cs="$char_sep" '{
  printf("%s ", $1);
  for (i = 2; i <= NF; ++i) {
    if ($i == ws) { printf(" "); }
    else { printf("%s%s", cs, $i); }
  }
  printf("\n");
}';
