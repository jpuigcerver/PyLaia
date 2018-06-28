#!/usr/bin/env bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

output_dir="";
help_message="
Usage: ${0##*/} [options] xml [xml ...]

Description:
  Convert XMLs in PAGE2010 format to PAGE2013.

Options:
  --output_dir : (type = string, default = \"$output_dir\")
                 Directory where output XML files will be written.
";
source "$SDIR/parse_options.inc.sh" || exit 1;
[ $# -lt 1 ] && echo "$help_message" >&2 && exit 1;

if [ -n "$output_dir" ]; then
  mkdir -p "$output_dir";
fi;

tmp="$(mktemp)";
while [ $# -gt 0 ]; do
  xmlstarlet tr "$SDIR/pagexml_2010_to_2013.xslt" "$1" \
    | sed 's| xmlns:_="[^"]*"||; s|pagecontent/2010-03-19|pagecontent/2013-07-15|g;' \
    | xmlstarlet ed -d //_:PlainText -d //@pcGtsId \
    > "$tmp";
  if [ -z "$output_dir" ]; then
    mv "$tmp" "$1";
  else
    mv "$tmp" "$output_dir/$(basename "$1")";
  fi;
  shift;
done;
