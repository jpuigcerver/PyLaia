#!/bin/bash
set -e;

beam=;
cache_size=100;
normalize=true;
use_qsub=false;
scale=1;
threads=1;
maxvmem=4G;
maxrt=4:00:00;
wdir=;
# Hidden option
merge=0;
swap=false;
help_message="
Usage: ${0##*/} [options] <query-scp> <word-scp> <out>

Options:
  -b BEAM
  -c CACHE_SIZE
  -n NORMALIZE
  -q USE_QSUB
  -s SCALE
  -t THREADS

Additional qsub options:
  -M MAX_VMEM
  -T MAX_RUNTIME
";
while [ "${1:0:1}" = "-" ]; do
  case "$1" in
    -b)
      beam="$2"; shift 2;
      ;;
    -c)
      cache_size="$2"; shift 2;
      ;;
    -n)
      normalize="$2"; shift 2;
      ;;
    -q)
      use_qsub="$2"; shift 2;
      ;;
    -s)
      scale="$2"; shift 2;
      ;;
    -t)
      threads="$2"; shift 2;
      ;;
    -M)
      maxvmem="$2"; shift 2;
      ;;
    -S)
      swap="$2"; shift 2;
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
    *)
      echo "Unknown option: \"$1\"" >&2 && exit 1;
      ;;
  esac;
done;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;
[ ! -s "$1" ] && echo "File \"$1\" does not exist!" >&2 && exit 1;
[ ! -s "$2" ] && echo "File \"$2\" does not exist!" >&2 && exit 1;

if [[ "$merge" -eq 0 && -z "$SGE_TASK_ID" && -f "$3" ]]; then
  msg="File \"$3\" already exists. Do you wish to overwrite it (y or n)? ";
  read -p "$msg" -n 1 -r; echo;
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    :
  else
    exit 0;
  fi;
fi;

# fst-compose-sum options
opts=(
  "--print-args=false"
  "--cache_size=$cache_size"
  "--normalize=$normalize"
  "--num_threads=$threads"
  "--scale=$scale"
);
[ -n "$beam" ] && opts+=("--beam=$beam");

if [[ -z "$SGE_TASK_ID" && "$merge" -eq 0 ]]; then
  if [[ "$use_qsub" = "true" ]]; then
    NQ="$(wc -l "$1" | cut -d\  -f1)";
    NW="$(wc -l "$2" | cut -d\  -f1)";

    if [[ "$NQ" -gt "$NW" ]]; then
        NT="$NQ";
        qscp="$1";
        wscp="$2";
        swap=false;
    else
        NT="$NW";
        qscp="$2";
        wscp="$1";
        swap=true;
    fi;

    [ -n "$wdir" ] || wdir="$(mktemp -d --tmpdir=$SCRATCH)";
    jid=$(qsub -terse -cwd -pe mp "$threads" \
        -t "1-$NT" -l "h_vmem=$maxvmem,h_rt=$maxrt" -j y -o "$wdir" \
        "${BASH_SOURCE[0]}" \
        -b "$beam" -c "$cache_size" -n "$normalize" -s "$scale" -t "$threads" \
        -W "$wdir" -S "$swap" "$qscp" "$wscp" "$3" |
        tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g');
    qsub -terse -cwd -l "h_vmem=1G,h_rt=$maxrt" -j y -o "$wdir" \
         -hold_jid "$jid" \
         "${BASH_SOURCE[0]}" -W "$wdir" -X "$1" "$2" "$3";
  else
    fst-compose-sum "${opts[@]}" "scp:$1" "scp:$2" |
    if [[ "${3:(-3)}" = ".gz" ]]; then
      gzip -9
    elif [[ "${3:(-3)}" = ".bz2" ]]; then
      bzip2 -9
    else
      cat
    fi > "$3";
  fi;
elif [[ -n "$SGE_TASK_ID" ]]; then
  [ ! -d "$wdir" ] && echo "Directory \"$wdir\" does not exist!" >&2 && exit 1;
  tmpscp="$(mktemp)";
  head -n$SGE_TASK_ID "$1" | tail -n1 > "$tmpscp";
  fst-compose-sum "${opts[@]}" "scp:$tmpscp" "scp:$2" |
  if [ "$swap" = true ]; then
    awk '{ t=$1; $1=$2; $2=t; print; }';
  else
    cat;
  fi > "$wdir/scores.${SGE_TASK_ID}.txt";
  rm "$tmpscp";
elif [[ "$merge" -eq 1 ]]; then
  [ ! -d "$wdir" ] && echo "Directory \"$wdir\" does not exist!" >&2 && exit 1;
  find "$wdir" -name "scores.*.txt" | xargs cat |
  if [[ "${3:(-3)}" = ".gz" ]]; then
    gzip -9
  elif [[ "${3:(-3)}" = ".bz2" ]]; then
    bzip2 -9
  else
    cat
  fi > "$3";
  rm -r "$wdir";
else
  echo "ERROR: Undefined behavior!" >&2 && exit 1;
fi;
