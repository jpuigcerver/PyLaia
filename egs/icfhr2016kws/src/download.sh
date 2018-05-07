#!/bin/bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";

mkdir -p data/images/{page,query};
mkdir -p data/images/word;
mkdir -p data/xmls;

BASEURL=https://www.prhlt.upv.es/contests/icfhr2016-kws/data;
DATASETS=(Botany Konzilsprotokolle);
EXPECTED_PAGES=(174 105);
EXPECTED_QUERIES=(150 200);
EXPECTED_WORDS=(3230 3534);
DOWNLOADED_PAGES=(
  "$(find data/images/page -name "b*.jpg" | wc -l)"
  "$(find data/images/page -name "k*.jpg" | wc -l)"
);
DOWNLOADED_QUERIES=(
  "$(find data/images/query -name "bq*.jpg" | wc -l)"
  "$(find data/images/query -name "kq*.jpg" | wc -l)"
);
DOWNLOADED_WORDS=(
  "$(find data/images/word -name "bw*.jpg" | wc -l)"
  "$(find data/images/word -name "kw*.jpg" | wc -l)"
);
d=0;
for d in 0 1; do
  ds="${DATASETS[d]}";
  # DOCUMENT PAGE IMAGES
  [ "${DOWNLOADED_PAGES[d]}" -eq "${EXPECTED_PAGES[d]}" ] || {
    # Download training pages
    for s in I II III; do
      [ -s data/${ds}_Train_${s}_PageImages.zip ] ||
      wget --no-check-certificate -P data -N \
	   ${BASEURL}/${ds}_Train_${s}_PageImages.zip;
      unzip -d data data/${ds}_Train_${s}_PageImages.zip;
      find data/${ds}_Train_${s}_PageImages -name "*.jpg" |
      xargs -n1 -I{} mv {} data/images/page/;
      rm -r data/${ds}_Train_${s}_PageImages;
      rm data/${ds}_Train_${s}_PageImages.zip;
    done;
    # Download document test pages
    [ -s data/${ds}_Test_PageImages.zip ] ||
    wget --no-check-certificate -P data -N \
	 ${BASEURL}/${ds}_Test_PageImages.zip;
    unzip -d data data/${ds}_Test_PageImages.zip;
    find data/${ds}_Test_PageImages -name "*.jpg" |
    xargs -n1 -I{} mv {} data/images/page/;
    rm -r data/${ds}_Test_PageImages;
    rm data/${ds}_Test_PageImages.zip;
  }
  # QUERY IMAGES
  [ "${DOWNLOADED_QUERIES[d]}" -eq "${EXPECTED_QUERIES[d]}" ] || {
    [ -s data/${ds}_Test_QryImages.zip ] ||
    wget --no-check-certificate -P data -N \
	 ${BASEURL}/${ds}_Test_QryImages.zip;
    unzip -d data data/${ds}_Test_QryImages.zip;
    find data/${ds}_Test_QryImages -name "*.jpg" |
      xargs -n1 -I{} mv {} data/images/query;
    rm -r data/${ds}_Test_QryImages;
    rm data/${ds}_Test_QryImages.zip;
  }
  # TEST WORD IMAGES
  [ "${DOWNLOADED_WORDS[d]}" -eq "${EXPECTED_WORDS[d]}" ] || {
    [ -s data/${ds}_Test_WordImages.zip ] ||
    wget --no-check-certificate -P data -N \
	 ${BASEURL}/${ds}_Test_WordImages.zip;
    unzip -d data data/${ds}_Test_WordImages.zip;
    find data/${ds}_Test_WordImages -name "*.jpg" |
      xargs -n1 -I{} mv {} data/images/word;
    rm -r data/${ds}_Test_WordImages;
    rm data/${ds}_Test_WordImages.zip;
  }
  # Download KWS Ground-Truth
  [ -s data/xmls/${ds}_Test_GT_SegBased_QbE.xml -a \
    -s data/xmls/${ds}_Test_GT_SegBased_QbS.xml -a \
    -s data/xmls/${ds}_Test_GT_SegFree_QbE.xml -a \
    -s data/xmls/${ds}_Test_GT_SegFree_QbS.xml ] || {
    wget --no-check-certificate -P data -N ${BASEURL}/${ds}_Test_GT.zip;
    unzip -d data/xmls data/${ds}_Test_GT.zip;
    rm data/${ds}_Test_GT.zip;
  }
  # Download training GT
  for s in I II III; do
    [ -s data/xmls/${ds}_Train_${s}_LL.xml -a \
      -s data/xmls/${ds}_Train_${s}_WL.xml -a \
      -s data/xmls/${ds}_Train_${s}_WL_CASE_INSENSITIVE.xml ] || {
      wget --no-check-certificate -P data -N \
	   ${BASEURL}/${ds}_Train_${s}_XML.zip;
      unzip -d data/xmls data/${ds}_Train_${s}_XML.zip;
      rm data/${ds}_Train_${s}_XML.zip;
    }
  done;
done;
