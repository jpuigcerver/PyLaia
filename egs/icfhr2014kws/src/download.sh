#!/usr/bin/env bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;

# Export PATH to utils
export PATH="$PWD/../utils:$PATH";

# Official data from the H-KWS 2014 competition
mkdir -p data/duth;
base_url=http://vc.ee.duth.gr/H-KWS2014/downloads/dataset;
base_url2=http://vc.ee.duth.gr/H-KWS2014/downloads/resources;
urls=(
  "$base_url/TRACK_I_Bentham_Dataset.7z"
  "$base_url/TRACK_I_Modern_Dataset.7z"
  "$base_url/TRACK_II_Bentham_Dataset.7z"
  "$base_url/TRACK_II_Modern_Dataset.7z"
  "$base_url/queries/ICFHR2014_TRACK_I_Bentham_Queries.7z"
  "$base_url/queries/ICFHR2014_TRACK_I_Modern_Queries.7z"
  "$base_url/queries/ICFHR2014_TRACK_II_Bentham_Queries.7z"
  "$base_url/queries/ICFHR2014_TRACK_II_Modern_Queries.7z"
  "$base_url2/TRACK_I_Bentham_ICFHR2014.RelevanceJudgements.xml.7z"
  "$base_url2/TRACK_I_Modern_ICFHR2014.RelevanceJudgements.xml.7z"
  "$base_url2/TRACK_II_Bentham_ICFHR2014.RelevanceJudgements.xml.7z"
  "$base_url2/TRACK_II_Modern_ICFHR2014.RelevanceJudgements.xml.7z"
  #"$base_url2/TRACK_I_Bentham_WordSegmentation.7z"
  #"$base_url2/TRACK_I_Modern_WordSegmentation.7z"
);
for url in ${urls[@]}; do
  file="${url##*/}";
  file="${file/.7z/}";
  # Download data, if it does not exist
  [ -s "data/duth/$file.7z" -o \
    -s "data/duth/$file" -o \
    -d "data/duth/$file" ] ||  wget -P data/duth "$url";
  # Extract data, if it does not exist
  [ -s "data/duth/$file" -o \
    -d "data/duth/$file" ] ||
  7z x -odata/duth "data/duth/$file.7z";
  rm -f "data/duth/$file.7z";
done;

# Download evaluation tool
[ -s data/duth/VCGEvalConsole.tar.gz -o \
  -d data/duth/VCGEvalConsole ] ||
wget -P data/duth \
  http://vc.ee.duth.gr/H-KWS2014/downloads/evaluation/VCGEvalConsole.tar.gz;
[ -d data/duth/VCGEvalConsole ] ||
tar zxf data/duth/VCGEvalConsole.tar.gz -C data/duth;

# Download training data from the ICFHR-2014 Competition on HTR.
mkdir -p data/prhlt;
base_url=http://www.transcriptorium.eu/~htrcontest/contestICFHR2014/public_html/HTRtS2014;
[ -s data/prhlt/contestHTRtS.tbz -o \
  -d data/prhlt/contestHTRtS ] ||
wget -P data/prhlt "$base_url/contestHTRtS.tbz";
[ -d data/prhlt/contestHTRtS ] || {
  tar xjf data/prhlt/contestHTRtS.tbz -C data/prhlt;
  rm -f data/prhlt/contestHTRtS.tbz;
  # Convert PAGE XML 2010 to 2013 format
  find data/prhlt/contestHTRtS/BenthamData/PAGE -name "*.xml" |
  xargs convert_page_xml_2010_to_2013.sh
}

# Download automatically segmented lines from test pages
base_url=https://www.prhlt.upv.es/~jpuigcerver/;
[ -s data/prhlt/icfhr2014kws_bentham_auto_segmented_lines_hpp.tar.gz -o \
  -d data/prhlt/icfhr2014kws_bentham_auto_segmented_lines_hpp ] ||
wget --no-check-certificate -P data/prhlt \
  "${base_url}/icfhr2014kws_bentham_auto_segmented_lines_hpp.tar.gz";
[ -d data/prhlt/icfhr2014kws_bentham_auto_segmented_lines_hpp ] || {
  tar zxf data/prhlt/icfhr2014kws_bentham_auto_segmented_lines_hpp.tar.gz \
      -C data/prhlt;
  rm -f data/prhlt/icfhr2014kws_bentham_auto_segmented_lines_hpp.tar.gz;
}
