#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

acoustic_scale=1;
beam=30;
determinize_lattice=true;
lattice_beam=15;
max_active=2147483647;
min_active=200;
qsub_opts="";
qsub_tasks=0;
retry_beam="";
symbol_table="";
threads=1;
# Hidden options used by qsub jobs
qsub_merge=true;
qsub_wdir="";
help_message="
Usage: ${0##*/} [options] <model> <HCL> <G> <loglike_scp> <lattice_wspec>

Arguments:
  model            : Kaldi's model containing the HMMs.
  HCL              : HCL transducer.
  G                : Grammar transducer.
  loglike_scp      : Kaldi's table script for the log-likelihoods.
  lattice_wspec    : Output lattice wspecifier.

Options:
  --acoustic_scale : (type = float, value = $acoustic_scale)
                     Scaling factor for acoustic likelihoos.
  --beam           : (type = float, value = $beam)
                     Decoding beam.
  --lattice_beam   : (type = float, value = $lattice_beam)
                     Lattice generation beam.
  --max_active     : (type = int, value = $max_active)
                     Decoder max active states.
  --min_active     : (type = int, value = $min_active)
                     Decoder minimum #active states.
  --qsub_opts      : (type = string, value = $qsub_opts)
                     Qsub options to parallelize in a cluster.
  --retry_beam     : (type = string, value = \"$retry_beam\")
                     List of beam to try when decoding fails.
  --symbol_table   : (type = string, value = \"$symbol_table\")
                     Symbol table used for debug the output of the decoding.
  --threads        : (type = int, value $threads)
                     When running locally, use this number of threads.
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -ne 5 ] &&
echo "ERROR: The program expected 5 arguments, but $# were given.

$help_message" >&2 && exit 1;

for f in "$1" "$2" "$3" "$4"; do
  [ ! -s "$f" ] &&
  echo "ERROR: File \"$f\" does not exist or is empty!" >&2 && exit 1;
done;

opts=(
  --acoustic-scale="$acoustic_scale"
  --beam="$beam"
  --determinize-lattice="$determinize_lattice"
  --lattice-beam="$lattice_beam"
  --max-active="$max_active"
  --min-active="$min_active"
  --retry-beam="$retry_beam"
  --word-symbol-table="$symbol_table"
);
opts2=(
  --acoustic_scale "$acoustic_scale"
  --beam "$beam"
  --determinize_lattice "$determinize_lattice"
  --lattice_beam "$lattice_beam"
  --max_active "$max_active"
  --min_active "$min_active"
  --retry_beam "$retry_beam"
  --symbol_table "$symbol_table"
  --qsub_tasks "$qsub_tasks"
);

if [[ -n "$SGE_TASK_ID" ]]; then
  [ -z "$qsub_wdir" ] && echo "ERROR: Empty working dir!" >&2 && exit 1;
  N=$(wc -l "$4" | cut -d\  -f1);
  D=$(echo "$N / $qsub_tasks" | bc);
  R=$(echo "$N % $qsub_tasks" | bc);
  if [ "$SGE_TASK_ID" -gt "$R" ]; then
    h="$[R * (D + 1) + (SGE_TASK_ID - R) * D]";
    t="$D";
  else
    h="$[SGE_TASK_ID * (D + 1)]";
    t="$[D + 1]";
  fi;
  egrep -s -q "^LOG \(.+\) Overall log-likelihood per frame" \
    "${qsub_wdir}/${SGE_TASK_ID}.log" ||
  latgen-lazylm-faster-mapped \
    "${opts[@]}" "$1" "$2" "$3" \
    "scp:head -n$h $4 | tail -n$t|" \
    "ark:${qsub_wdir}/${SGE_TASK_ID}.ark" \
    2> "${qsub_wdir}/${SGE_TASK_ID}.log";
elif [[ -n "$SGE_CELL" && "$qsub_merge" = true ]]; then
  [ -z "$qsub_wdir" ] && echo "ERROR: Empty working dir!" >&2 && exit 1;
  completed=$(egrep -s -l "^LOG \(.+\) Overall log-likelihood" \
    "$qsub_wdir"/*.log | wc -l);
  if [[ "$completed" != "$qsub_tasks" ]]; then
    echo "ERROR: Some tasks were not completed, check: $qsub_wdir!" >&2 &&
    exit 1;
  fi;
  find "$qsub_wdir" -name "*.ark" | sort -V |
  xargs -n1 -I{} lattice-copy ark:{} ark:- |
  lattice-copy ark:- "$5";
  rm -r "$qsub_wdir";
elif [[ "$qsub_tasks" -gt 0 ]]; then
  if [ -z "$qsub_wdir" ]; then qsub_wdir="$(mktemp -d --tmpdir="$SCRATCH")"; fi;
  opts2+=(--qsub_wdir "$qsub_wdir");
  jid1=$(cat <<EOF | qsub | tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g'
#\$ -N ${0##*/}
#\$ -j y -o "$qsub_wdir"
#\$ -t 1-${qsub_tasks}
#\$ $qsub_opts
#\$ -terse
#\$ -cwd
set -e;
${BASH_SOURCE[0]} --qsub_merge false $(printf '%q ' "${opts2[@]}" "$@")
EOF
  );
  [ "$qsub_merge" = false ] ||
  jid2=$(cat <<EOF | qsub | tail -n1 | sed -r 's|.[0-9]+-[0-9]+:[0-9]+$||g'
#\$ -N ${0##*/}
#\$ -j y -o "$qsub_wdir"
#\$ $qsub_opts
#\$ -terse
#\$ -cwd
#\$ -hold_jid $jid1
set -e;
${BASH_SOURCE[0]} --qsub_merge true $(printf '%q ' "${opts2[@]}" "$@")
EOF
  );
  echo "Launched jobs: $jid1 $jid2" >&2;
else
  if [ "$threads" -gt 1 ]; then
    N=$(wc -l "$4" | cut -d\  -f1);
    D=$(echo "$N / $threads" | bc);
    R=$(echo "$N % $threads" | bc);
    tmp=();
    for t in $(seq "$threads"); do
      if [ "$t" -gt "$R" ]; then
	h="$[R * (D + 1) + (t - R) * D]";
	t="$D";
      else
	h="$[t * (D + 1)]";
	t="$[D + 1]";
      fi;
      tmp+=("$(mktemp)");
      latgen-lazylm-faster-mapped "${opts[@]}" "$1" "$2" "$3" \
				  "scp:head -n$h $4 | tail -n$t|" \
				  "ark:${tmp[-1]}" &
    done;
    wait || exit 1;
    lattice-copy "ark:cat ${tmp[*]}|" "$5";
    rm -f "${tmp[@]}";
  else
    latgen-lazylm-faster-mapped "${opts[@]}" "$1" "$2" "$3" "scp:$4" "$5";
  fi;
fi;
