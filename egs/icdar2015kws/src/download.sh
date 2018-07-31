#!/bin/bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";


BASEURL=http://transcriptorium.eu/~icdar15kws/data;

mkdir -p data/images/train_lines data/lang/lines/word;
mkdir -p data/images/{pages,words}/{te,va};

# Download TRAIN data
[ -s data/lang/lines/word/tr_original.txt -a \
  -s data/images/train_lines/002_080_001_01_01.png -a \
  -s data/images/train_lines/116_649_001_03_24.png ] || {
  wget --no-check-certificate -P data -N \
       ${BASEURL}/Train_Lines_Images.zip \
       ${BASEURL}/Train_Lines_Transcriptions.zip;
  [ -d data/Train_Lines_Images ] || unzip -d data data/Train_Lines_Images.zip;
  find data/Train_Lines_Images -name "*.png" |
  xargs -n1 -I{} mv {} data/images/train_lines;

  [ -d data/Train_Lines_Transcriptions ] ||
  unzip -d data data/Train_Lines_Transcriptions.zip;
  find data/Train_Lines_Transcriptions -name "*.txt" |
  xargs awk '{
    n = split(FILENAME, A, "/");
    bn = substr(A[n], 1, length(A[n]) - 4);
    print bn, $0;
  }' | sort -V > data/lang/lines/word/tr_original.txt;
  rm -r data/Train_Lines_Transcriptions* data/Train_Lines_Images*;
}

# Download VALID and TEST data word images
QIMG=(Validation_QueryByExample Test_Query_Images);
QKWS=(Validation_QueryByString Test_Query_Keywords);
EXPECTED_QIMG=(95 1421);
DOWNLOADED_QIMG=(
  $(find data/images/words/va -name "q*.jpg" | wc -l)
  $(find data/images/words/te -name "q*.jpg" | wc -l)
);
EXPECTED_WIMG=(3234 15419);
DOWNLOADED_WIMG=(
  $(find data/images/words/va -name "w*.jpg" | wc -l)
  $(find data/images/words/te -name "w*.jpg" | wc -l)
);
EXPECTED_PIMG=(10 70);
DOWNLOADED_PIMG=(
  $(find data/images/pages/va -name "*.jpg" | wc -l)
  $(find data/images/pages/te -name "*.jpg" | wc -l)
);
k=0;
for p in Validation Test; do
  pn=$(echo ${p:0:2} | tr '[:upper:]' '[:lower:]');
  [[ ( ${DOWNLOADED_WIMG[k]} -eq ${EXPECTED_WIMG[k]} ) && \
     ( ${DOWNLOADED_QIMG[k]} -eq ${EXPECTED_QIMG[k]} ) && \
     ( ${DOWNLOADED_PIMG[k]} -eq ${EXPECTED_PIMG[k]} ) ]] ||
  wget --no-check-certificate -P data -N \
       ${BASEURL}/${QIMG[k]}.zip \
       ${BASEURL}/${QKWS[k]}.lst \
       ${BASEURL}/${p}_Document_Images.zip \
       ${BASEURL}/${p}_SegmentedWord_Images.zip \
       ${BASEURL}/${p}_GT_TrackI_AssignmentA.inf \
       ${BASEURL}/${p}_GT_TrackI_AssignmentB.inf \
       ${BASEURL}/${p}_GT_TrackII_AssignmentA.inf \
       ${BASEURL}/${p}_GT_TrackII_AssignmentB.inf;

  # Extract document pages
  [ ${DOWNLOADED_PIMG[k]} -eq ${EXPECTED_PIMG[k]} ] || {
    [ -d data/${p}_Document_Images ] ||
    unzip -d data data/${p}_Document_Images.zip;
    find data/${p}_Document_Images -name "*.jpg" |
    xargs -n1 -P$(nproc) -I{} mv {} data/images/pages/$pn;
    rm -r data/${p}_Document_Images data/${p}_Document_Images.zip;
  }

  # Extract document segmented words
  [ ${DOWNLOADED_WIMG[k]} -eq ${EXPECTED_WIMG[k]} ] || {
    [ -d data/${p}_SegmentedWord_Images ] ||
    unzip -d data data/${p}_SegmentedWord_Images.zip;
    find data/${p}_SegmentedWord_Images -name "*.jpg" |
    xargs -n1 -P$(nproc) -I{} mv {} data/images/words/$pn;
    rm -r data/${p}_SegmentedWord_Images data/${p}_SegmentedWord_Images.zip;
  }

  # Extract query images
  [ ${DOWNLOADED_QIMG[k]} -eq ${EXPECTED_QIMG[k]} ] || {
    [ -d data/${QIMG[k]} ] ||
    unzip -d data data/${QIMG[k]}.zip;
    find data/${QIMG[k]} -name "*.jpg" |
    xargs -n1 -P$(nproc) -I{} mv {} data/images/words/$pn;
    rm -r data/${QIMG[k]} data/${QIMG[k]}.zip;
  }

  k=$[k+1];
done;
