#!/bin/bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

mkdir -p data;

wget -N -P data ftp://scruffy.cvl.tuwien.ac.at/staff/database/cvl-database/cvl-database-1-1.zip;
[ -d data/cvl-database-1-1 ] || unzip -d data data/cvl-database-1-1.zip;
