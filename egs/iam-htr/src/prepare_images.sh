#!/bin/bash
set -e;

# Directory where the script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/src" != "$SDIR" ] && \
echo "Please, run this script from the experiment top directory!" >&2 && \
exit 1;

source ../utils/functions_check.inc.sh || exit 1;
source ../utils/functions_parallel.inc.sh || exit 1;

num_parallel="$(get_num_cores)";
overwrite=false;
resize_height=128;
help_message="
Usage: ${0##*/} [options]

Options:
  --nun_parallel  : (type = integer, default = $num_parallel)
                    Number of parallel processes to process the images.
  --overwrite     : (type = boolean, default = $overwrite)
                    Overwrite previously created files.
  --resize_height : (type = integer, default = $resize_height)
                    Resize the line and sentence images to this given height.
";
source ../utils/parse_options.inc.sh || exit 1;

check_imgtxtenh || exit 1;
check_imagemagick || exit 1;

function process_image () {
  local bn="$(basename "$1" .png)";
  # Process image
  [ "$overwrite" = false -a -s "${img_dir}/$bn.jpg" ] ||
  imgtxtenh -d 118.110 "$1" png:- |
  convert png:- -deskew 40% \
    -bordercolor white -border 5 -trim \
    -bordercolor white -border 20x0 +repage \
    -strip "${img_dir}/$bn.jpg" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  # Resize image to a fixed height, keep aspect ratio.
  [ "$overwrite" = false -a -s "${img_resize_dir}/$bn.jpg" ] ||
  convert "${img_dir}/$bn.jpg" -resize "x${resize_height}" +repage \
    -strip "${img_resize_dir}/$bn.jpg" ||
  { echo "ERROR: Processing image $1" >&2 && return 1; }
  return 0;
}


tmpd="$(mktemp -d)";
partitions=(lines sentences);
expected_count=(13353 16752 115320);
bkg_pids=();
for p in $(seq ${#partitions[@]}); do
  partition="${partitions[p-1]}";

  # Check that the partition was downloaded and extracted!
  [ ! -d "data/original/$partition" ] &&
  echo "ERROR: No data found for partition \"$partition\"!" >&2 &&
  exit 1;


  img_dir="data/imgs/${partition}";
  img_resize_dir="data/imgs/${partition}_h${resize_height}";
  mkdir -p "${img_dir}" "${img_resize_dir}";
  actual_count="$(find "$img_resize_dir" -name "*.jpg" | wc -l)";
  [[ "$overwrite" = false && "$actual_count" -eq "${expected_count[p-1]}" ]] || {
    # Enhance images with Mauricio's tool, deskew the line, crop white borders
    # and resize to the given height.
    for f in $(find "data/original/$partition" -name "*.png"); do
      process_image "$f" &> "$tmpd/${#bkg_pids[@]}" &
      bkg_pids+=("$!");
      [ "${#bkg_pids[@]}" -lt "$num_parallel" ] ||
      { wait_jobs --log_dir "$tmpd" "${bkg_pids[@]}" && bkg_pids=(); } || exit 1;
    done;
  }
done;
wait_jobs --log_dir "$tmpd"  "${bkg_pids[@]}" || exit 1;
rm -rf "$tmpd";
exit 0;
