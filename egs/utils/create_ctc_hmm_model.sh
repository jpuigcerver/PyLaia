#!/bin/bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

eps="<eps>";
ctc="<ctc>";
dregex="^#.+";
overwrite=false;
ploop=0.6;
pctc=0.5;
verbose=false;
help_message="
Usage: ${0##*/} [options] symbols_map output_model output_tree

Arguments:
  symbols_map  : File containing the mapping from string to integer IDs of
                 ALL symbols (must include: all characters, the CTC blank
                 symbol and the dummy symbol).
  output_model : Path to the output model.
  output_tree  : Path to the output tree.

Options:
  --ctc        : (type = string, default = \"$ctc\")
                 String representing the CTC blank symbol in the symbols map.
  --dregex     : (type = regex, default = \"$dregex\")
                 Regular expression for disambiguation symbols in the
                 symbols map.
  --eps        : (type = string, default = \"$eps\")
                 String representing the epsilon symbol in the symbols map.
  --overwrite  : (type = boolean, default = $overwrite)
                 If true, overwrite output files, even if it is not needed.
  --pctc       : (type = float, default = $pctc)
                 Transition probability to the CTC states.
  --ploop      : (type = float, default = $ploop)
                 Self-loop probability in the emitting states.
  --verbose    : (type = bolean, default = $verbose)
                 If true, show information about the output model and tree.
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -ne 3 ] && echo "$help_message" >&2 && exit 1;

symbs_all="$1";
model="$2";
tree="$3";

# Create output directories.
mkdir -p "$(dirname "$model")" "$(dirname "$tree")";

# This is the list of all the actual characters (no epsilon, no ctc,
# and no disambiguation symbols).
char_ids=( $(gawk -v ctc="$ctc" -v eps="$eps" -v dr="$dregex" \
  '$1!=eps && $1!=ctc && $1!~dr{print $2;}' $symbs_all | sort -n) );
# This is the integer ID for the epsilon symbol. MUST BE 0
eps_id="$(gawk -v eps="$eps" '$1 == eps{print $2}' $symbs_all)";
# This is the integer ID for the CTC symbol.
ctc_id="$(gawk -v ctc="$ctc" '$1 == ctc{print $2}' $symbs_all)";
# Maximum symbol ID, excluding disambiguation and epsilon
max_id=$(printf "%d\n" ${char_ids[@]} $ctc_id | sort -nr | head -n1);

# Check that <eps> has integer ID equal to 0.
[ -z "$eps_id" ] && eps_id=0;
[ "$eps_id" -ne 0 ] && echo "ERROR: Integer ID 0 must be used for $eps!" &&
exit 1;

# Map from pdf to symbol ID.
# Maps each column in Laia's confidence matrix (i.e. the PDF) to the
# corresponding symbol (ctc + characters).
pdf2sym=( $(printf "%s\n" "${char_ids[@]}" "$ctc_id" | sort -n) );
# Map from symbol to pdf ID.
# Maps each symbol (ctc + characters) to its column in Laia's confidence matrix
# (i.e. the PDF). Symbols with -1 mean that they don't have any associated pdf
# (e.g. epsilon, and other missing IDs).
sym2pdf=();
for i in $(seq 0 $max_id); do sym2pdf[$i]=-1; done; # Initialize array
for i in ${!pdf2sym[@]}; do sym2pdf[${pdf2sym[i]}]=$i; done;

pnloop="$(printf "%g" "$(echo "1.0 - $ploop" | bc -l)")";
ptctc="$(printf "%g" "$(echo "$pctc * (1.0 - $ploop)" | bc -l)")";
ptfin="$(printf "%g" "$(echo "(1.0 - $pctc) * (1.0 - $ploop)" | bc -l)")";

tmpdir="$(mktemp -d)";
cat <<EOF > "$tmpdir/topo"
<Topology>
<TopologyEntry>
<ForPhones>
$ctc_id
</ForPhones>
  <State> 0 <PdfClass> 0 <Transition> 0 $ploop <Transition> 1 $pnloop </State>
  <State> 1 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones>
${char_ids[@]}
</ForPhones>
  <State> 0 <PdfClass> 0 <Transition> 0 $ploop <Transition> 1 $ptctc <Transition> 2 $ptfin </State>
  <State> 1 <PdfClass> 1 <Transition> 1 $ploop <Transition> 2 $pnloop </State>
  <State> 2 </State>
</TopologyEntry>
</Topology>
EOF


# Initialize LogProbs and acoustic parameters. Acoustic parameters don't matter
# since they are not used in practice.
gmm-init-mono --binary=false --print-args=false \
  "$tmpdir/topo" "${#pdf2sym[@]}" "$tmpdir/auxMdl" "$tmpdir/auxTree";

# Fix transition model so that pdfs are shared across different blank states.
cat <<EOF > "$tmpdir/model"
<TransitionModel>
$(cat $tmpdir/topo)
<Triples> $[2 * ${#char_ids[@]} + 1]
$ctc_id 0 ${sym2pdf[ctc_id]}
$(for c in ${char_ids[@]}; do
  echo $c 0 ${sym2pdf[c]}
  echo $c 1 ${sym2pdf[ctc_id]}
done)
</Triples>
$(sed -n "/<LogProbs>/,/<\/LogProbs>/p" $tmpdir/auxMdl)
</TransitionModel>
EOF

# Acoustic model. Meaningless, the only important thing is that numpdfs is
# equal to the number of characters + 1 (ctc).
cat <<EOF >> "$tmpdir/model"
<DIMENSION> 1 <NUMPDFS> ${#pdf2sym[@]}
$(for ((i=1;i<=${#pdf2sym[@]};++i)); do
  echo "<DiagGMM>";
  echo "<GCONSTS> [ 0 ]";
  echo "<WEIGHTS> [ 1 ]";
  echo "<MEANS_INVVARS> [ 1 ]";
  echo "<INV_VARS> [ 1 ]";
  echo "</DiagGMM>";
done)
EOF

# Custom context dependency tree.
cat <<EOF > "$tmpdir/tree"
ContextDependency 1 0 ToPdf TE 0 ${#sym2pdf[@]} (
$(for s in $(seq 1 ${#sym2pdf[@]}); do
  ((--s)); p=${sym2pdf[$s]};
  if [[ $p -eq -1 ]]; then
    echo "NULL";               ## The symbol does not have an associated HMM.
  elif [[ $s -eq $ctc_id ]]; then
    echo "TE -1 1 ( CE $p )";  ## Dummy HMM.
  else
    echo "TE -1 2 ( CE $p CE ${sym2pdf[$ctc_id]} )";
  fi;
done)
)
EndContextDependency
EOF

# Copy temporal files to the output locations, only if needed.
if [[ "$overwrite" = true || ! -s "$model" ]] ||
  ! cmp -s "$tmpdir/model" "$model"; then
  mv "$tmpdir/model" "$model";
fi;
if [[ "$overwrite" = true || ! -s "$tree" ]] ||
  ! cmp -s "$tmpdir/tree" "$tree"; then
  mv "$tmpdir/tree" "$tree";
fi;

# Remove temporal dir;
rm -rf "$tmpdir";

# Show model info.
[ "$verbose" = true ] && {
  echo "CTC-HMM model info:" >&2;
  gmm-info --print-args=false "$model" >&2;
  tree-info --print-args=false "$tree" >&2;
}

exit 0;
