#!/usr/bin/env bash
set -e;

height=0;
offset=0.0;
scale=1.0;
dims_file="";
help_message="
Usage: ${0##*/} [options] index_type index_file [index_file ...]

Description:

Arguments:
  index_type  : (type = string)
                Type of the index files (either \"segment\" or \"position\").
  index_file  : (type = string)
                Filepath of an index file (in plain text).

Options:
  -h  : (type = int, default=$height)
        Assume that the height of the images is this.
        Note: If this is zero, -D must be specified.
  -s  : (type = float, default = $scale)
        Scale start and end frames by this factor.
  -x  : (type = float, default = $offset)
        Apply this offset to the start and end frames (applied after scale).
  -D  : (type = int, default = $dims_file)
        Read from this file the number of frames and image height of each sample,
        otherwise it assumes that the length is found in the second column of the
        index file.
";
while [ "${1:0:1}" = "-" ]; do
  case "$1" in
    -h)
      height="$2";
      shift 2 || { echo "ERROR: Missing height!" >&2 && exit 1; }
      ;;
    -s)
      scale="$2";
      shift 2 || { echo "ERROR: Missing scale!" >&2 && exit 1; }
    ;;
    -x)
      offset="$2";
      shift 2 || { echo "ERROR: Missing offset!" >&2 && exit 1; }
    ;;
    -D)
      dims_file="$2";
      shift 2 || { echo "ERROR: Missing dimensions file!" >&2 && exit 1; }
    ;;
    *)
      echo "ERROR: Unknown option \"$1\"!" >&2 && exit 1;
  esac;
done;
[ $# -lt 2 ] && echo "$help_message" >&2 && exit 1;

index_type="$1"; shift;

if [ -n "$dims_file" ]; then
  [ "$(echo "$height > 0" | bc)" -eq 1 ] &&
  echo "WARNING: You specified -h and -D, fixed height will be ignored" >&2 &&
  exit 1;

  join -1 1 <(sort -V "$dims_file") <(sort -V "$@");
else
  [ "$(echo "$height > 0" | bc)" -ne 1 ] &&
  echo "ERROR: You must specify either -h or -D" >&2 &&
  exit 1;

  awk -v height="$height" '{
    printf("%s %d %d", $1, $2, height);
    for (i = 3; i <= NF; ++i) { printf(" %s", $i); }
    printf("\n");
  }' "$@";
fi |
awk -v index_type="$index_type" -v scale="$scale" -v offset="$offset" '{
  printf("%s", $1);
  w=$2; h=$3;

  if (index_type == "position") {
    step = 6;
  } else {
    step = 5;
  }

  for (i = 4; i <= NF; i += step) {
    if (i > 4) printf(" ;");
    if (index_type == "position") {
      x0 = $(i+2) * scale + offset;
      x1 = $(i+3) * scale + offset;
      y0 = 0; y1 = h;
      printf(" %s %d %g,%g %g,%g %g,%g %g,%g %.8g",
             $i, $(i+1), x0, y0, x1, y0, x1, y1, x0, y1, $(i+4));
    } else {
      x0 = $(i+1) * scale + offset;
      x1 = $(i+2) * scale + offset;
      y0 = 0; y1 = h;
      printf(" %s %g,%g %g,%g %g,%g %g,%g %.8g",
             $i, x0, y0, x1, y0, x1, y1, x0, y1, $(i+3));
    }
  }
}';