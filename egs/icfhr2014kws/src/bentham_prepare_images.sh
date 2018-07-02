#!/usr/bin/env bash
set -e;
export LC_NUMERIC=C;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;

check_textFeats || exit 1;
check_all_programs convert find identify || exit 1;
check_all_dirs data/prhlt/contestHTRtS/BenthamData/Images/Lines \
	           data/duth/ICFHR2014_TRACK_II_Bentham_Queries/images || exit 1;

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
# If image height > fix_height, scale to height 80.
function fix_image_height () {
  [ $# -ne 3 ] && \
  echo "Usage: fix_image_height <fix_height> <input_img> <output_img>" >&2 && \
  return 1;

  h=$(identify -format '%h' "$2") || return 1;
  if [ "$h" -lt "$1" ]; then
    convert -gravity center -extent x80 +repage -strip "$2" "$3" || return 1;
  else
	convert -resize x80 +repage -strip "$2" "$3" || return 1;
  fi;
  return 0;
}

# Number of expected lines in the dataset.
ne=10613;

# Process text line images
mkdir -p data/bentham/imgs/lines;
nr=$(find data/bentham/imgs/lines -name "*.png" | wc -l);
[ -d data/bentham/imgs/lines -a "$nr" -eq "$ne" ] || {
  mkdir -p data/bentham/imgs/lines;
  # Create link to JPG images in the directory containing the PAGE images,
  # this is needed by the textFeats tool.
  ln -frs data/prhlt/contestHTRtS/BenthamData/Images/Pages/*.jpg \
          data/prhlt/contestHTRtS/BenthamData/PAGE/;
  textFeats \
    --cfg="$cfg" \
    --outdir=data/bentham/imgs/lines \
    --overwrite=true \
    --savexml=data/bentham/imgs/lines \
    --threads=$(nproc) \
    data/prhlt/contestHTRtS/BenthamData/PAGE/*.xml;

  find data/bentham/imgs/lines -name "*.xml" |
  xargs xmlstarlet sel -t -m '//_:TextLine' \
                       -v '../../@imageFilename' -o ' ' \
                       -v @id -o " " -m '_:Property[@key="fpgram"]' \
                       -v @value -n |
  sed -r 's|^([0-9_]+)\.jpg |\1.|g' |
  sort -V > data/bentham/imgs/lines/fpgrams.txt;
}

# Fix height of the line images.
mkdir -p data/bentham/imgs/lines_h80;
nr=$(find data/bentham/imgs/lines_h80 -name "*.png" | wc -l);
[ "$ne" -eq "$nr" ] || {
  n=0;
  for f in $(find data/bentham/imgs/lines -name "*.png"); do
    ( fix_image_height 80 "$f" "${f/lines/lines_h80}" || exit 1; ) &
    ((++n));
    [ "$n" -eq "$(nproc)" ] && { wait; n=1; }
  done;
  wait;

  # Fix parallelograms
  find data/bentham/imgs/lines -name "*.png" |
  xargs identify -format "%f %h %w\n" |
  sed -r 's|^(.+)\.png |\1 |g' | sort -V |
  gawk '
  function get_coord(s) {
    if (!match(s, /([0-9.]+),([0-9.]+)/, m)) {
        print "ERROR: Wrong x,y coordinate at line" NR > "/dev/stderr";
        exit 1;
    }
    return m;
  }
  {
    scale = 1;
    l_offset = 0;
    r_offset = 0;
    t_offset = 0;
    b_offset = 0;

    if ($2 > 80) {
      scale = $2 / 80.0;
    } else {
      t_offset =  (80 - $2) / 2.0;
      b_offset = -(80 - $2) / 2.0;
    }
    print $1, scale, l_offset, r_offset, t_offset, b_offset;
  }' |
  sort -V > data/bentham/imgs/lines_h80/resize_info.txt;
}


# Process query images
mkdir -p data/bentham/imgs/queries;
ne=$(find data/duth/ICFHR2014_TRACK_II_Bentham_Queries/images -name "*.png" | wc -l);
nr=$(find data/bentham/imgs/queries -name "*.png" | wc -l);
[ "$ne" -eq "$nr" ] || {
  find data/duth/ICFHR2014_TRACK_II_Bentham_Queries/images -name "*.png" |
  xargs textFeats \
	--cfg="$cfg" \
	--outdir=data/bentham/imgs/queries \
	--overwrite=true \
	--threads=$(nproc);
}

# Fix height of the query images.
mkdir -p data/bentham/imgs/queries_h80;
[ "$(find data/bentham/imgs/queries -name "*.png" | wc -l)" -eq \
  "$(find data/bentham/imgs/queries_h80 -name "*.png" | wc -l)" ] || {
  for f in $(find data/bentham/imgs/queries -name "*.png"); do
    fix_image_height 80 "$f" "${f/queries/queries_h80}" || exit 1;
  done;
}

# Extract bounding box of the biggest text region in each page.
# This is used to create an offset on the bounding boxes of the words detected
# in the original pages, since the ICFHR2014KWS competition extracted the
# main text region from each test page.
[ -s data/bentham/imgs/page_biggest_regions.txt ] ||
find data/bentham/imgs/lines -name "*.xml" |
xargs xmlstarlet sel -t -m '//_:TextRegion' -v '../@imageFilename' -o ' ' \
      -m '_:Coords' -v '@points' -n  |
awk '{
  xmin = 100000; xmax = -100000; ymin = 100000; ymax = -100000;
  for (i=2; i <= NF; ++i) {
    split($i, xy, ",");
    if (xmin > xy[1]) xmin = xy[1];
    if (xmax < xy[1]) xmax = xy[1];
    if (ymin > xy[2]) ymin = xy[2];
    if (ymax < xy[2]) ymax = xy[2];
  }
  w = (xmax - xmin);
  h = (ymax - ymin);
  area = w * h;
  if (area > MAX_AREA[$1]) {
    MAX_AREA[$1] = area;
    OFFSET_X[$1] = xmin;
    OFFSET_Y[$1] = ymin;
    WIDTH[$1]  = w;
    HEIGHT[$1] = h;
  }
}END{
  for (page in MAX_AREA) {
    print page, OFFSET_X[page], OFFSET_Y[page], WIDTH[page], HEIGHT[page];
  }
}' |
sed -r 's|^([0-9_]+)\.jpg|\1|g' |
sort -V > data/bentham/imgs/page_biggest_regions.txt;

mkdir -p data/bentham/imgs/auto_segmented_lines;
nr=$(find data/bentham/imgs/auto_segmented_lines -name "*.png" | wc -l);
[ "$nr" -eq 1053 ] || {
  textFeats \
    --cfg="$cfg" \
    --outdir=data/bentham/imgs/auto_segmented_lines \
    --overwrite=true \
    --savexml=data/bentham/imgs/auto_segmented_lines \
    --threads=$(nproc) \
    data/prhlt/icfhr2014kws_bentham_auto_segmented_lines_hpp/*.xml;

  find data/bentham/imgs/auto_segmented_lines -name "*.xml" |
  xargs xmlstarlet sel -t -m '//_:TextLine' \
                       -v '../../@imageFilename' -o ' ' \
                       -v @id -o " " -m '_:Property[@key="fpgram"]' \
                       -v @value -n |
  sed -r 's|^([0-9_]+)\.jpg |\1.|g' |
  sort -V > data/bentham/imgs/auto_segmented_lines/fpgrams.txt;
}


# Fix height of the line images.
mkdir -p data/bentham/imgs/auto_segmented_lines_h80;
nr=$(find data/bentham/imgs/auto_segmented_lines_h80 -name "*.png" | wc -l);
[ "$nr" -eq 1053 ] || {
  n=0;
  for f in $(find data/bentham/imgs/auto_segmented_lines -name "*.png"); do
    ( fix_image_height 80 "$f" "${f/lines/lines_h80}" || exit 1; ) &
    ((++n));
    [ "$n" -eq "$(nproc)" ] && { wait; n=1; }
  done;
  wait;

  # Fix parallelograms
  find data/bentham/imgs/auto_segmented_lines -name "*.png" |
  xargs identify -format "%f %h %w\n" |
  sed -r 's|^(.+)\.png |\1 |g' | sort -V |
  gawk '
  function get_coord(s) {
    if (!match(s, /([0-9.]+),([0-9.]+)/, m)) {
        print "ERROR: Wrong x,y coordinate at line" NR > "/dev/stderr";
        exit 1;
    }
    return m;
  }
  {
    scale = 1;
    l_offset = 0;
    r_offset = 0;
    t_offset = 0;
    b_offset = 0;

    if ($2 > 80) {
      scale = $2 / 80.0;
    } else {
      t_offset =  (80 - $2) / 2.0;
      b_offset = -(80 - $2) / 2.0;
    }
    print $1, scale, l_offset, r_offset, t_offset, b_offset;
  }' |
  sort -V > data/bentham/imgs/auto_segmented_lines_h80/resize_info.txt;
}

# Remove temp config file
rm -f "$cfg";
