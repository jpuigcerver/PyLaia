#!/bin/bash
set -e;

# Directory where the script is placed.
source "../utils/functions_check.inc.sh" || exit 1;

iam_password=;
iam_username=;
help_message="
Usage: ${0##*/} [options]

Options:
  --iam_password : (type = string, default = \"$iam_password\")
                   Password for the IAM server.
  --iam_username : (type = string, default = \"$iam_username\")
                   Username for the IAM server.
";
source "../utils/parse_options.inc.sh" || exit 1;

check_all_programs find tar wget || exit 1;

# Utility function to download files from IAM.
function download_url () {
  [ -z "$iam_username" -o -z "$iam_password" ] && \
    echo "Please, use the --iam_username and --iam_password options to " \
	 "download the database from the IAM servers." >&2 && return 1;
  wget -P data/original --user="$iam_username" --password="$iam_password" \
       "$1" ||
  { echo "ERROR: Failed downloading $1!" >&2 && return 1; }
  return 0;
}


partitions=(lines sentences words);
imgs_url=(
  http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz
  http://www.fki.inf.unibe.ch/DBs/iamDB/data/sentences/sentences.tgz
  http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz
);
expected_count=(13353 16752 115320);

for p in $(seq ${#partitions[@]}); do
  partition="${partitions[p-1]}";
  mkdir -p "data/original/${partition}";

  actual_count="$(find data/original/$partition -name "*.png" | wc -l)";
  [ "${actual_count}" -eq "${expected_count[p-1]}" ] || {
    # Download images from IAM.
    [ -s "data/original/${partition}.tgz" ] ||
    download_url "${imgs_url[p-1]}" || exit 1;
    # Extract images.
    {
      tmpd="$(mktemp -d)";
      tar zxf "data/original/${partition}.tgz" -C "$tmpd";
      find "$tmpd" -name "*.png" | xargs -I{} mv {} "data/original/$partition";
      rm -rf "$tmpd";
      # Note: Remove tar file not needed anymore.
      rm "data/original/${partition}.tgz";
    } ||
    { echo "ERROR: Failed extracting file data/origina/${partition}.tgz" >&2 && exit 1; }
  }
done;

# Download ascii files.
url="http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz";
[ -s data/original/ascii.tgz ] || download_url "$url" || exit 1;

# Untar ascii files.
[ -s data/original/lines.txt -a -s data/original/sentences.txt \
  -a -s data/original/forms.txt -a -s data/original/words.txt ] ||
tar zxf data/original/ascii.tgz -C data/original ||
( echo "ERROR: Failed extracting IAM ascii files." >&2 && exit 1 );

# Dowload splits typically used in HTR papers.
[ -s data/splits/graves/te.lst -a \
  -s data/splits/graves/tr.lst -a \
  -s data/splits/graves/va.lst -a \
  -s data/splits/original/te.lst -a \
  -s data/splits/original/tr.lst -a \
  -s data/splits/original/va1.lst -a \
  -s data/splits/original/va2.lst -a \
  -s data/splits/pham/te.lst -a \
  -s data/splits/pham/tr.lst -a \
  -s data/splits/pham/va.lst -a \
  -s data/splits/puigcerver/te.lst -a \
  -s data/splits/puigcerver/tr.lst -a \
  -s data/splits/puigcerver/va.lst ] || {
  [ -s data/iam_splits.tar.gz ] ||
  wget --no-check-certificate -P data \
       https://www.prhlt.upv.es/~jpuigcerver/iam_splits.tar.gz ||
  { echo "ERROR: Failed downloading IAM splits files!" >&2 && return 1; }
  tar zxf data/iam_splits.tar.gz -C data;
  rm data/iam_splits.tar.gz;
}

rm -rf "$tmpd";
exit 0;
