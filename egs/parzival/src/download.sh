#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR/..";

[ ! -f "../utils/parse_options.inc.sh" ] &&
echo "Missing \"$SDIR/../../utils/parse_options.inc.sh\" file!" >&2 && exit 1;

iam_pass=;
iam_user=;
help_message="
Usage: ${0##*/} [options]

Options:
  --iam_pass   : (type = string, default = \"$iam_pass\")
                 Password for the IAM server.
  --iam_user   : (type = string, default = \"$iam_user\")
                 Username for the IAM server.
";
source "../utils/parse_options.inc.sh" || exit 1;
[ $# -ne 0 ] && echo "$help_message" >&2 && exit 1;

# Utility function to download files from IAM.
function download_url () {
  listopt=;
  if [ "$1" = "-i" ]; then listopt=-i; shift 1; fi;
  [ -z "$iam_user" -o -z "$iam_pass" ] && \
    echo "Please, use the --iam_user and --iam_pass options to download" \
	 "the database from the IAM servers." >&2 && return 1;
  wget -N -P "$2" --user="$iam_user" --password="$iam_pass" $listopt "$1" ||
  { echo "ERROR: Failed downloading $1!" >&2 && return 1; }
  return 0;
}

mkdir -p data;
url="http://www.fki.inf.unibe.ch/DBs/iamHistDB/data/parzivaldb-v1.0.zip";
[ -s data/parzivaldb-v1.0.zip -o -d data/parzivaldb-v1.0 ] ||
download_url "$url" data;
[ -d data/parzivaldb-v1.0 ] ||
unzip -d data data/parzivaldb-v1.0.zip;
rm -rf data/__MACOSX;
find data/parzivaldb-v1.0 -name ".DS_Store" | xargs -r rm -f;
