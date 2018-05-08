#!/bin/bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

mkdir -p data/images/words;

NUM_PNG="$(find data/images/words -name "*.png" | wc -l)";
[ "$NUM_PNG" -eq 99904 ] && exit 0;

URL=ftp://scruffy.cvl.tuwien.ac.at/staff/database/cvl-database/cvl-database-1-1.zip;
NUM_TIF="$(find data/cvl-database-1-1/ -name "*.tif" | wc -l)";
[ "$NUM_TIF" -eq 114981 ] || {
  wget -N -P data "$URL";
  unzip -d data data/cvl-database-1-1.zip;
}
