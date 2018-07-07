#!/bin/bash
set -e;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/src" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/src/parse_options.inc.sh" ] && \
    echo "Missing $(pwd)/src/parse_options.inc.sh file!" >&2 && exit 1;

nproc="$(nproc)";
overwrite=false;
partition=lines;
help_message="
Usage: ${0##*/} [options]

Options:
  --nproc      : (type = integer, default = $nproc)
                 Use this number of concurrent processes to prepare the images.
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
  --partition  : (type = string, default = \"$partition\")
                 Select the \"lines\" or \"sentences\" partition.
                 Note: Typically the \"lines\" partition is used.
";
source "$(pwd)/src/parse_options.inc.sh" || exit 1;

which imgtxtenh &> /dev/null ||
( echo "ERROR: Install https://github.com/mauvilsa/imgtxtenh" >&2 && exit 1 );
which convert &> /dev/null ||
( echo "ERROR: Install ImageMagick's convert" >&2 && exit 1 );

tmpd="$(mktemp -d)";

function process_image () {
  local bn="$(basename "$1" .png)";
  # Process image
  [ "$overwrite" = false -a -s "data/imgs/$partition/$bn.jpg" ] ||
  imgtxtenh -d 118.110 "$1" png:- |
  convert png:- -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    -strip "data/imgs/$partition/$bn.jpg" ||
  ( echo "ERROR: Processing image $1" >&2 && return 1 );
  # Resize image
  [ "$overwrite" = false -a -s "data/imgs/${partition}_h128/$bn.jpg" ] ||
  convert "data/imgs/$partition/$bn.jpg" -resize "x128" +repage \
    -strip "data/imgs/${partition}_h128/$bn.jpg" ||
  ( echo "ERROR: Processing image $1" >&2 && return 1 );
  return 0;
}

function wait_jobs () {
  local n=0;
  while [ $# -gt 0 ]; do
    if ! wait "$1"; then
      echo "Failed image processing:" >&2 && cat "$tmpd/$n" >&2 && return 1
    fi;
    shift 1; ((++n));
  done;
  return 0;
}

# Check that the partition was downloaded and extracted!
[ ! -d "data/original/$partition" ] &&
echo "ERROR: No data found for partition \"$partition\"!" >&2 &&
exit 1;

# Enhance images with Mauricio's tool, deskew the line, crop white borders
# and resize to the given height.
mkdir -p data/imgs/${partition}{,_h128};
bkg_pids=();
for f in data/original/$partition/*.png; do
  process_image "$f" &> "$tmpd/${#bkg_pids[@]}" &
  bkg_pids+=("$!");
  [ "${#bkg_pids[@]}" -lt "$nproc" ] ||
  { wait_jobs "${bkg_pids[@]}" && bkg_pids=(); } ||
  exit 1;
done;
wait_jobs "${bkg_pids[@]}" || exit 1;


mkdir -p data/lists/$partition/{aachen,original};
for c in aachen original; do
  for f in data/part/$partition/$c/*.lst; do
    bn=$(basename "$f" .lst);
    [ -s "data/lists/$partition/$c/$bn.lst" ] ||
    gawk '{ print $1 }' "$f" \
    > "data/lists/$partition/$c/$bn.lst" || exit 1;
  done;
done;

rm -rf "$tmpd";
exit 0;
