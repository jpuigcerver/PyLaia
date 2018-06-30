#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

URL=http://ciir.cs.umass.edu/downloads/gw/gw_20p_wannot.tgz;
[ -s data/gw_20p_wannot.tgz ] || wget -O data/gw_20p_wannot.tgz "$URL";

[ -d data/gw_20p_wannot -a -s data/gw_20p_wannot/3090309.tif ] ||
tar zxf data/gw_20p_wannot.tgz -C data;
