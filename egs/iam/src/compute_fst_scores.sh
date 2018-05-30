#!/bin/bash
set -e;
export LC_NUMERIC=C;

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
help_message="
Usage: ${0##*/} [options] <query-scp> <distractors-scp> <out>

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
[[ -n "$2" && ( ! -s "$2" ) ]] &&
echo "File \"$2\" does not exist!" >&2 && exit 1;

if [[ "$merge" -eq 0 && -z "$SGE_TASK_ID" && -f "$3" ]]; then
  msg="File \"$3\" already exists. Do you wish to overwrite it (y or n)? ";
  read -p "$msg" -n 1 -r; echo;
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    :
  else
    exit 0;
  fi;
fi;

function pairwise_scp () {
  [ $# -ne 1 ] && echo "Usage: pairwise_scp <scp>" >&2 && return 1;
  tmp="$(mktemp -u)";
  awk -v tmp="$tmp" 'BEGIN{
    N = 0;
  }{
    SCP[N++] = $0;
  }END{
    for (i = 1; i <= N; ++i) {
      print SCP[i - 1] > tmp "." i ".1";
      for (j = i + 1; j <= N; ++j) {
        print SCP[j - 1] > tmp "." i ".2";
      }
      close(tmp "." i ".1");
      close(tmp "." i ".2");
    }
  }' "$1";
  echo "$tmp";
  return 0;
}

function memusg_wrap () {
  if which memusg &> /dev/null; then
    memusg "$@";
  else
    "$@";
  fi;
}

# fst-compose-sum options
opts=(
  "--print-args=false"
  "--cache-size=$cache_size"
  "--normalize=$normalize"
  "--num-threads=$threads"
  "--scale=$scale"
);
[ -n "$beam" ] && opts+=("--beam=$beam");

if [[ -z "$SGE_TASK_ID" && "$merge" -eq 0 ]]; then
  if [[ "$use_qsub" = "true" ]]; then
    [ -n "$wdir" ] || wdir="$(mktemp -d --tmpdir=$SCRATCH)";
    NT="$(wc -l "$1" | cut -d\  -f1)";
    jid=$(qsub -terse -cwd -pe mp "$threads" \
        -t "1-$NT" -l "h_vmem=$maxvmem,h_rt=$maxrt" -j y -o "$wdir" \
        "${BASH_SOURCE[0]}" \
        -b "$beam" -c "$cache_size" -n "$normalize" -s "$scale" -t "$threads" \
        -W "$wdir" "$1" "$2" "$3" |
        tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g');
    jid2=$(qsub -terse -cwd -l "h_vmem=2G,h_rt=$maxrt" -j y -o "$wdir" \
         -hold_jid "$jid" \
         "${BASH_SOURCE[0]}" -W "$wdir" -X "$1" "$2" "$3" |
         tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g');
    echo "Launched jobs: $jid $jid2" >&2;
  else
    mkdir -p "$(dirname "$3")";
    tmp="$(pairwise_scp "$1")";
    {
      NQ="$(wc -l "$1" | cut -d\  -f1)";
      for i in $(seq $[NQ - 1]); do
          fst-compose-sum "${opts[@]}" "scp:$tmp.$i.1" "scp:$tmp.$i.2";
      done | awk '{ print; print $2, $1, $3; }';
      if [ -n "$2" ]; then
          fst-compose-sum "${opts[@]}" "scp:$1" "scp:$2";
      fi;
    } |
    if [[ "${3:(-3)}" = ".gz" ]]; then
      gzip -9
    elif [[ "${3:(-3)}" = ".bz2" ]]; then
      bzip2 -9
    else
      cat
    fi > "$3";
  fi;
elif [[ -n "$SGE_TASK_ID" && "$merge" -eq 0 ]]; then
  [ ! -d "$wdir" ] && echo "Directory \"$wdir\" does not exist!" >&2 && exit 1;
  tmpscp1="$(mktemp)";
  head "-n$SGE_TASK_ID" "$1" | tail -n1 > "$tmpscp1";
  tmpscp2="$(mktemp)";
  tail "-n+$[SGE_TASK_ID + 1]" "$1" > "$tmpscp2";
  {
    if [ "$(wc -l "$tmpscp2" | cut -d\  -f1)" -gt 0 ]; then
      memusg_wrap fst-compose-sum "${opts[@]}" "scp:$tmpscp1" "scp:$tmpscp2" |
      awk '{ print; print $2, $1, $3; }';
    fi;
    if [ -n "$2" ]; then
      memusg_wrap fst-compose-sum "${opts[@]}" "scp:$tmpscp1" "scp:$2";
    fi;
  } > "$wdir/scores.${SGE_TASK_ID}.txt";
  rm "$tmpscp1" "$tmpscp2";
elif [[ "$merge" -eq 1 ]]; then
  [ ! -d "$wdir" ] && echo "Directory \"$wdir\" does not exist!" >&2 && exit 1;
  mkdir -p "$(dirname "$3")";
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
