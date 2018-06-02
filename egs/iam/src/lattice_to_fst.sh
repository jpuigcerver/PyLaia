#!/bin/bash
set -e;
export LC_NUMERIC=C;
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

acoustic_scale=1;
beam=inf;
graph_scale=1;
model="";
help_message="
Usage: ${0##*/} <query_list> <distractor_list> <ark> [<ark> ...] <output_scp>

Options:
  --acoustic_scale    : (type = float, value = $acoustic_scale)
  --beam              : (type = float, value = $beam)
  --graph_scale       : (type = float, value = $graph_scale)
  --model             : (type = string, value = \"$model\")
                        Model to extract the characters from the lattice.
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -lt 4 ] && echo "$help_message" >&2 && exit 1;


for f in "$1" "$2"; do
  [ ! -f "$f" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
done;
qlist="$1";
dlist="$2";
shift 2;

files=();
while [ $# -gt 1 ]; do
  [ ! -f "$1" ] && echo "ERROR: File \"$f\" does not exist!" >&2 && exit 1;
  files+=("$1");
  shift;
done;

if [ "${1:(-4)}" != ".scp" ]; then
  echo "ERROR: Output file must end with .scp!" >&2 && exit 1;
fi;

mkdir -p "$(dirname "$1")";
oscp="$1";
oark="${oscp/.scp/.ark}";
if [ ${oscp:(-8)} = ".fst.scp" ]; then
  qscp="${oscp/.fst.scp/_q.fst.scp}";
  dscp="${oscp/.fst.scp/_d.fst.scp}";
else
  qscp="${oscp/.scp/_q.scp}";
  dscp="${oscp/.scp/_d.scp}";
fi;

if [ -s "$oscp" -a -s "$oark" ]; then
  msg="Output \"$oscp\" already exists. Continue (c) or abort (a)?";
  read -p "$msg " -n 1 -r; echo;
  if [[ $REPLY =~ ^[Cc]$ ]]; then
    :
  else
    exit 0;
  fi;
fi;

if [ -n "$model" ]; then
  [ ! -s "$model" ] &&
  echo "ERROR: Model \"$model\" does not exist!" >&2 &&
  exit 1;
fi;

for f in "${files[@]}"; do
  if [ "${f:(-4)}" = .ark ]; then
    cat "$f";
  elif [ "${f:(-7)}" = .ark.gz ]; then
    zcat "$f";
  elif [ "${f:(-8)}" = .ark.bz2 ]; then
    bzcat "$f";
  else
    echo "WARNING: Ignored input \"$f\"" >&2;
  fi;
done |
if [ -n "$model" ]; then
  lattice-to-phone-lattice --replace-words=false "$model" ark:- ark:- |
  lattice-project --project-output=false ark:- ark:-;
else
  cat;
fi |
if [ "$beam" != "inf" ]; then
  lattice-prune --beam="$beam" ark:- ark:-;
else
  cat;
fi |
lattice-to-fst \
  --acoustic-scale="$acoustic_scale" --lm-scale="$graph_scale" \
  ark:- "ark,scp:$oark,$oscp";

join -1 1 \
  <(cut -d\  -f1 "$qlist" | sort) \
  <(sort "$oscp") > "$qscp";

join -1 1 \
  <(cut -d\  -f1 "$dlist" | sort) \
  <(sort "$oscp") > "$dscp";
