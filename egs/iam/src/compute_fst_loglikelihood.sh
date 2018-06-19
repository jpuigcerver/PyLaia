#!/bin/bash
set -e;
export LC_NUMERIC=C;
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

function txt2voc () {
  sed -r 's|\s+| |g;s|^ +||g;s| +$||g' |
  awk -v bos="$bos" -v eos="$eos" '{
    if (bos != "") {
      w = bos;
      s = 1;
    } else {
      w = $1;
      s = 2;
    }
    for (i = s; i <= NF; ++i) { w = w " " $i; }
    if (eos != "") {
      w = w" "eos;
    }
    print w;
  }' | sort | uniq -c || return 1;
  return 0;
}

function txt2fst () {
  awk 'BEGIN{
    ns = 0;
    make_final_start = 0;
  }{
    if (NF == 0) {
      make_final_start = 1;
      next;
    }

    print 0, ++ns, $1;
    for (i = 2; i <= NF; ++i) {
      print ns, ++ns, $i;
    }
    print ns;
  }END{
    if (make_final_start) {
      print 0;
    }
  }' | fstcompile --acceptor --isymbols="$1" |
       fstarcsort --sort_type=olabel || return 1;
  return 0;
}

bos="";
delta=0.000001;
eos="";
extra_txt="";
nshortest=0;
syms="";
help_message="
Usage: ${0##*/} [options] <fst> <txt> [<txt> ...]

Arguments:
  fst:
  txt:

Options:
  --bos
  --delta
  --eos
  --extra_txt
  --nshortest
  --syms
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -lt 2 ] && echo "$help_message" >&2 && exit 1;

G="$1";
shift 1;

voc="$(mktemp)";
cat "$@" | txt2voc > "$voc";

ifst="$(mktemp)";
awk '{ for (i=2; i<= NF; ++i) printf("%s ", $i); printf("\n"); }' "$voc" |
txt2fst "$syms" | fstdeterminize | fstminimize |
fstcompose - "$G" |
fstprint | fstcompile --arc_type=log |
fstrmepsilon --delta="$delta" |
fstdeterminize --delta="$delta" |
fstminimize --delta="$delta" |
fstarcsort  --sort_type=ilabel |
fstprint > "$ifst";

if [ "$nshortest" -lt 1 -a -z "$extra_txt" ]; then
  cost_offset="$(fstprint "$G" | fstcompile --arc_type=log |
                 fstshortestdistance --delta="$delta" --reverse=true |
                 head -n1 | awk '{ print $2 }')";
  echo "INFO: Cost offset in G is $cost_offset" >&2;
else
  cost_offset=$( (\
    fstcompile --arc_type=log < "$ifst" | \
      fstshortestdistance --reverse=true | \
      head -n1 | awk '{print $2}';

    if [ -n "$extra_txt" ]; then
      txt2voc < "$extra_txt" |
      awk '{ for (i=2; i<= NF; ++i) printf("%s ", $i); printf("\n"); }' |
      txt2fst "$syms" | fstdeterminize | fstminimize |
      fstcompose - "$G" |
      fstprint | fstcompile --arc_type=log |
      fstshortestdistance --reverse=true |
      head -n1 | awk '{print $2}';
    fi;

    if [ "$nshortest" -gt 0 ]; then
      fstdifference "$G" <(awk '{
                           if (NF == 5 || NF == 2) { $NF = 0; }
                             print;
                           }' "$ifst" | fstcompile) |
      fstshortestpath --nshortest="$nshortest" |
      fstprint | fstcompile --arc_type=log |
      fstshortestdistance --reverse=true |
      head -n1 | awk '{print $2}';
    fi;
  ) | python -c "
from __future__ import print_function
import math
import sys

def logadd(a, b):
  if a < b:
    a, b = b, a
  return a + math.log1p(math.exp(b - a))

total_cost = float('inf')
for line in sys.stdin:
  cost = float(line)
  total_cost = -logadd(-total_cost, -cost)

print(total_cost)
");
fi;

if [ -n "$syms" ]; then
  "$SDIR/sym2int.pl" -f 2- "$syms" "$voc";
else
  cat "$voc";
fi |
python -c "
from __future__ import division
from __future__ import print_function
import sys

# Load transducer
G = {}
F = {}
with open('$ifst', 'r') as fst:
  for line in fst:
    line = line.split()
    if len(line) == 1:
      F[int(line[0])] = 0.0
    elif len(line) == 2:
      F[int(line[0])] = float(line[1])
    elif len(line) == 4 or len(line) == 5:
      w = float(line[4]) if len(line) == 5 else 0.0
      s0, s1 = int(line[0]), int(line[1])
      if s0 in G:
        G[s0].append((s1, line[2], line[3], w))
      else:
        G[s0] = [(s1, line[2], line[3], w)]


def find_arc(s0, l):
  if s0 not in G:
    return None
  for (s1, l1, l2, w) in G[s0]:
    if l1 == l:
      return (s1, l1, l2, w)
  return None

# Compute average cost (-loglikelihood)
total_cost = 0
total_count = 0
total_oov = 0
for line in sys.stdin:
  line = line.split()
  count = int(line[0])
  cs = 0
  cost = 0
  for lbl in line[1:]:
    arc = find_arc(cs, lbl)
    if arc is None:
      cost = float('inf')
      break
    else:
      cs = arc[0]
      cost += arc[-1]
  if cost < float('inf'):
    if cs in F:
      cost += F[cs]

  if cost < float('inf'):
    total_cost += count * (cost - $cost_offset)
    total_count += count
  else:
    total_oov += count

total_words = (total_count + total_oov)
perc_oov = total_oov / total_words
print('INFO: Total words: {:d}'.format(total_words), file=sys.stderr)
print('INFO: OOV {:d} ({:.2%})'.format(total_oov, perc_oov), file=sys.stderr)
print(total_cost / total_count)
";

rm -f "$voc" "$ifst";
exit 0;
