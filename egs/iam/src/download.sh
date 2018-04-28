#!/bin/bash
set -e;

# Directory where the prepare.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)/src" != "$SDIR" ] && \
    echo "Please, run this script from the experiment top directory!" >&2 && \
    exit 1;
[ ! -f "$(pwd)/src/parse_options.inc.sh" ] &&
echo "Missing $(pwd)/src/parse_options.inc.sh file!" >&2 && exit 1;

iam_pass=;
iam_user=;
help_message="
Usage: ${0##*/} [options] <partition>

Options:
  --iam_pass   : (type = string, default = \"$iam_pass\")
                 Password for the IAM server.
  --iam_user   : (type = string, default = \"$iam_user\")
                 Username for the IAM server.
";
source "$(pwd)/src/parse_options.inc.sh" || exit 1;
[ $# -ne 1 ] && echo "$help_message" >&2 && exit 1;

# Utility function to download files from IAM.
function download_url () {
  listopt=;
  if [ "$1" = "-i" ]; then listopt=-i; shift 1; fi;
  [ -z "$iam_user" -o -z "$iam_pass" ] && \
    echo "Please, use the --iam_user and --iam_pass options to download" \
	 "the database from the IAM servers." >&2 && return 1;
  wget -P "$2" --user="$iam_user" --password="$iam_pass" $listopt "$1" ||
  { echo "ERROR: Failed downloading $1!" >&2 && return 1; }
  return 0;
}

mkdir -p data/original/ascii;
if [[ ( ! -s data/original/ascii/forms.txt ) ||
      ( ! -s data/original/ascii/lines.txt ) ||
      ( ! -s data/original/ascii/sentences.txt ) ||
      ( ! -s data/original/ascii/words.txt ) ]]; then
  # Download ascii files.
  url="http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz";
  [ -s data/original/ascii/ascii.tgz ] ||
  download_url "$url" data/original/ascii;
  tar zxf data/original/ascii/ascii.tgz -C data/original/ascii;
  rm data/original/ascii/ascii.tgz;
fi;

case "$1" in
  forms)
    mkdir -p data/original/forms;
    mapfile -t forms \
        <<< "$(awk '$1 !~ /^#/{ print $1; }' data/original/ascii/forms.txt)";
    url=http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/;
    if [[ ( ! -s "data/original/forms/${forms[0]}.png" ) ||
          ( ! -s "data/original/forms/${forms[-1]}.png" ) ]]; then
      download_url \
	-i <(awk -v url="$url" '$1 !~ /^#/{ print url""$1".png"; }' \
	     data/original/ascii/forms.txt) \
	data/original/forms;
    fi;
    ;;
  lines)
    mkdir -p data/original/lines;
    url="http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz";
    if [[ ( ! -s data/original/lines/a01/a01-000u/a01-000u-00.png ) ||
          ( ! -s data/original/lines/r06/r06-143/r06-143-04.png ) ]]; then
      [ -s data/original/lines/lines.tgz ] ||
      download_url "$url" data/original/lines;
      tar zxf data/original/lines/lines.tgz -C data/original/lines;
      rm data/original/lines/lines.tgz;
    fi;
    ;;
  sentences)
    mkdir -p data/original/sentences;
    url="http://www.fki.inf.unibe.ch/DBs/iamDB/data/sentences/sentences.tgz";
    if [[ ( ! -s data/original/sentences/a01/a01-000u/a01-000u-s00-00.png ) ||
          ( ! -s data/original/sentences/r06/r06-143/r06-143-s04-01.png ) ]];
    then
      [ -s data/original/sentences/sentences.tgz ] ||
      download_url "$url" data/original/sentences;
      tar zxf data/original/sentences/sentences.tgz -C data/original/sentences;
      rm data/original/sentences/sentences.tgz;
    fi;
    ;;
  words)
    mkdir -p data/original/words;
    url="http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz";
    if [[ ( ! -s data/original/words/a01/a01-000u/a01-000u-00-00.png ) ||
          ( ! -s data/original/words/r06/r06-143/r06-143-04-10.png ) ]]; then
      [ -s data/original/words/words.tgz ] ||
      download_url "$url" data/original/words;
      tar zxf data/original/words/words.tgz -C data/original/words;
    fi;
    ;;
  *)
    echo "ERROR: Unknown partition \"$1\"!" >&2 && exit 1;
esac;
