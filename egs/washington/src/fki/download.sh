#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/../..";
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";

overwrite=false;
help_message="
Usage: ${0##*/} [options]
Options:
  --overwrite  : (type = boolean, default = $overwrite)
                 Overwrite previously created files.
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

mkdir -p data/fki;

## Download lines images from FKI.
[ "$overwrite" = false -a -s data/fki/washingtondb-v1.0.zip ] || {
  [ -z "$FKI_USER" -o -z "$FKI_PASSWORD" ] &&
  echo "Please, set the FKI_USER and FKI_PASSWORD variables to download the" \
       "Washington database from the FKI servers." >&2 && exit 1;
  wget -P data/fki --user="$FKI_USER" --password="$FKI_PASSWORD" \
       http://www.fki.inf.unibe.ch/DBs/iamHistDB/data/washingtondb-v1.0.zip;
}

## Unzip dataset.
[ "$overwrite" = false -a -d data/fki/washingtondb-v1.0 ] ||
( cd data/fki && unzip -uqq washingtondb-v1.0.zip && \
  rm -rf __MACOSX && cd - &> /dev/null ) ||
exit 1;
