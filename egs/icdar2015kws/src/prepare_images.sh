#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

export LC_NUMERIC=C;
export PATH="$PWD/../utils:$PATH";

check_textFeats || exit 1;
check_all_programs convert find identify || exit 1;

height=80;
help_message="
Usage: ${0##*/} [options]

Options:
  --height : (type = int, default = $height)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

cfg="$(mktemp)";
cat <<EOF > "$cfg"
TextFeatExtractor: {
  verbose    = false;
  // Whether to do automatic desloping of the text
  deslope    = true;
  // Whether to do automatic deslanting of the text
  deslant    = true;
  // Type of feature to extract, either "dotm" or "raw"
  type       = "raw";
  // Output features format, either "htk", "ascii" or "img"
  format     = "img";
  // Whether to do contrast stretching
  stretch    = true;
  // Window size in pixels for local enhancement
  enh_win    = 30;
  // Sauvola enhancement parameter
  enh_prm    = 0.1;
  // 3 independent enhancements, each in a color channel
  //enh_prm   = [ 0.05, 0.2, 0.5 ];
  // Normalize image heights
  normheight = 0;
  normxheight = 0;
  // Global line vertical moment normalization
  momentnorm = true;
  // Whether to compute the features parallelograms
  fpgram     = true;
  // Whether to compute the features surrounding polygon
  fcontour   = true;
  //fcontour_dilate = 0;
  // Padding in pixels to add to the left and right
  padding    = 10;
}
EOF

# If image height < fix_height, pad with white.
# If image height > fix_height, scale.
function fix_image_height () {
  [ $# -ne 3 ] && \
  echo "Usage: fix_image_height <fix_height> <input_img> <output_img>" >&2 && \
  return 1;

  h=$(identify -format '%h' "$2") || return 1;
  if [ "$h" -lt "$1" ]; then
    convert -gravity center -extent "x$1" +repage -strip "$2" "$3" || return 1;
  else
	convert -resize "x$1" +repage -strip "$2" "$3" || return 1;
  fi;
  return 0;
}

#####################################################################
## 1. Clean training text line images with textFeats
#####################################################################
mkdir -p data/images/train_lines_proc;
[ "$(find data/images/train_lines -name "*.png" | wc -l)" -eq \
  "$(find data/images/train_lines_proc -name "*.png" | wc -l)" ] ||
find data/images/train_lines -name "*.png" |
xargs textFeats \
      --cfg="$cfg" \
      --outdir=data/images/train_lines_proc \
      --overwrite=true \
      --threads=$(nproc);


#####################################################################
## 2. Resize training text line images to a fixed height
#####################################################################
mkdir -p data/images/train_lines_proc_h${height};
[ "$(find data/images/train_lines -name "*.png" | wc -l)" -eq \
  "$(find data/images/train_lines_proc_h${height} -name "*.png" | wc -l)" ] || {
  n=0;
  for f in $(find data/images/train_lines_proc -name "*.png"); do
    ( fix_image_height "${height}" "$f" "${f/proc/proc_h${height}}" || exit 1; ) &
    ((++n));
    [ "$n" -eq "$(nproc)" ] && { wait; n=1; }
  done;
  wait;
}
