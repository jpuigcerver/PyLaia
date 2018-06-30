#!/bin/bash
set -e;
export LC_NUMERIC=C;

use_qsub=false;
maxvmem=4G;
maxrt=4:00:00;
threads=1;
wdir=;
# Hidden option
merge=0;
workload=20;
help_message="
Usage: ${0##*/} [options] <scp> <voc-fst> <fst-wspecifier>

Options:
  -q USE_QSUB
  -t THREADS

Additional qsub options:
  -L MIN_WORKLOAD
  -M MAX_VMEM
  -T MAX_RUNTIME
";

while [ "${1:0:1}" = "-" ]; do
  case "$1" in
    -q)
      use_qsub="$2"; shift 2;
      ;;
    -t)
      threads="$2"; shift 2;
      ;;
    -L)
      workload="$2"; shift 2;
      ;;
    -M)
      maxvmem="$2"; shift 2;
      ;;
    -T)
      maxrt="$2"; shift 2;
      ;;
    -W)
      wdir="$2"; shift 2;
      ;;
    -X)
      merge=1; shift 1;
      ;;
    --)
      shift;
      break;
      ;;
    *);
      echo "Unknown option: \"$1\"" >&2 && exit 1;
      ;;
  esac;
done;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;
[ ! -s "$1" ] && echo "File \"$1\" does not exist!" >&2 && exit 1;

if [[ -z "$SGE_TASK_ID" && "$merge" -eq 0 ]]; then
  if [[ "$use_qsub" = "true" ]]; then
    [ -n "$wdir" ] || wdir="$(mktemp -d --tmpdir=$SCRATCH)";
    N="$(wc -l "$1" | cut -d\  -f1)";
    NT="$(echo "$N / ($workload * $threads)" | bc)";
    if [ "$NT" -lt 1 ]; then NT=1; fi;
    echo $N $NT;
    jid=$(qsub -terse -cwd -pe mp "$threads" \
        -t "1-$NT" -l "h_vmem=$maxvmem,h_rt=$maxrt" -j y -o "$wdir" \
        "${BASH_SOURCE[0]}" \
        -t "$threads" -L "$workload" -W "$wdir" "$1" "$2" "$3" |
        tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g');
    jid2=$(qsub -terse -cwd -l "h_vmem=4G,h_rt=$maxrt" -j y -o "$wdir" \
         -hold_jid "$jid" \
         "${BASH_SOURCE[0]}" -L "$workload" -W "$wdir" -X "$1" "$2" "$3" |
         tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g');
    echo "Launched jobs: $jid $jid2" >&2;
  else
    fst-compose "--num-threads=$threads" "scp:$1" "$2" ark:- |
    fst-determinizestar --use-log ark:- ark:- |
    fst-minimize ark:- "$3";
  fi;
elif [[ -n "$SGE_TASK_ID" && "$merge" -eq 0 ]]; then
  [ ! -d "$wdir" ] && echo "Directory \"$wdir\" does not exist!" >&2 && exit 1;
  N="$(wc -l "$1" | cut -d\  -f1)";
  NT="$(echo "$N / ($workload * $threads)" | bc)";
  if [ "$NT" -lt 1 ]; then NT=1; fi;

  NL="$(echo "$N / $NT" | bc)";
  RL="$(echo "$N % $NT" | bc)";
  if [[ "$SGE_TASK_ID" -le "$RL" ]]; then
    h="$[SGE_TASK_ID * (NL + 1)]";
    t="$[NL + 1]";
  else
    h="$[RL * (NL + 1) + (SGE_TASK_ID - RL) * NL]";
    t="$NL";
  fi;

  fst-compose "--num-threads=$threads" \
	      <(head "-n$h" "$1" | tail "-n$t") "$2" ark:- |
  fst-determinizestar --use-log ark:- ark:- |
  fst-minimize ark:- "ark:$wdir/fst.${SGE_TASK_ID}.ark";
elif [[ "$merge" -eq 1 ]]; then
  [ ! -d "$wdir" ] && echo "Directory \"$wdir\" does not exist!" >&2 && exit 1;
  find "$wdir" -name "fst.*.txt" | sort -V | xargs cat |
  fstcopy --print-args=false ark:- "$3";
  #rm -r "$wdir";
else
  echo "ERROR: Undefined behavior!" >&2 && exit 1;
fi;
