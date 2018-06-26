#!/bin/bash
set -e;

for d in data/prhlt/contestHTRtS/BenthamData/Images/Lines \
	 data/duth/ICFHR2014_TRACK_II_Bentham_Queries/images; do
  [ ! -d "$d" ] && echo "Directory \"$d\" does not exist!" >&2 && exit 1;
done;

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
  fpgram     = false;
  // Whether to compute the features surrounding polygon
  fcontour   = true;
  //fcontour_dilate = 0;
  // Padding in pixels to add to the left and right
  padding    = 10;
}
EOF

# Process text line images
mkdir -p data/bentham/imgs/lines;
ne=$(find data/prhlt/contestHTRtS/BenthamData/Images/Lines -name "*.png"|wc -l);
nr=$(find data/bentham/imgs/lines -name "*.png" | wc -l);
[ "$ne" -eq "$nr" ] || {
  find data/prhlt/contestHTRtS/BenthamData/Images/Lines -name "*.png" |
  xargs textFeats \
	--cfg="$cfg" \
	--threads=$(nproc) \
	--outdir=data/bentham/imgs/lines \
	--overwrite=true \
	--savexml=data/bentham/imgs/lines;
}

# If height < 80, pad with white.
# If height > 80, scale to height 80.
mkdir -p data/bentham/imgs/lines_h80;
nr=$(find data/bentham/imgs/lines_h80 -name "*.png" | wc -l);
[ "$ne" -eq "$nr" ] || {
  n=0;
  for f in $(find data/bentham/imgs/lines -name "*.png"); do
    (
      h=$(identify -format '%h' "$f");
      if [ "$h" -lt 80 ]; then
	convert -gravity center -extent x80 +repage -strip \
		"$f" "${f/lines/lines_h80}";
      else
	convert -resize x80 +repage -strip \
		"$f" "${f/lines/lines_h80}";
      fi;
    ) &
    ((++n));
    [ "$n" -eq "$(nproc)" ] && { wait; n=1; }
  done;
  wait;
}

# Process query images
mkdir -p data/bentham/imgs/queries;
ne=$(find data/duth/ICFHR2014_TRACK_II_Bentham_Queries/images -name "*.png" | wc -l);
nr=$(find data/bentham/imgs/queries -name "*.png" | wc -l);
[ "$ne" -eq "$nr" ] || {
  find data/duth/ICFHR2014_TRACK_II_Bentham_Queries/images -name "*.png" |
  xargs textFeats \
	--cfg="$cfg" \
	--threads=$(nproc) \
	--outdir=data/bentham/imgs/queries \
	--overwrite=true \
	--savexml=data/bentham/imgs/queries;
}

mkdir -p data/bentham/imgs/queries_h80;
for f in $(find data/bentham/imgs/queries -name "*.png"); do
  convert -resize x80 +repage -strip "$f" "${f/queries/queries_h80}";
done;

# Remove temp config file
rm -f "$cfg";
