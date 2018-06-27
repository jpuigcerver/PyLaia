#!/bin/bash
set -e;

help_message="
Usage: ${0##*/} <dim> <sym_beg> <sym_end> <mat-rspecifier> <mat-wspecifier>

Description:
  Add boundary frames at the start and/or end of each sample.

  The symbols to add at each frame are given by 'sym_beg' and 'sym_end', which
  are strings containing integers. Each symbol must be 0 <= sym < 'dim'.
";
[ $# -ne 5 ] && echo "$help_message" >&2 && exit 1;

copy-matrix --print-args=false "$4" ark,t:- |
awk -v DIM="$1" -v SBEG="$2" -v SEND="$3" '
function create_frame(dim, ind) {
  s = "";
  for (i = 0; i < dim; ++i) {
    if (i == ind) { s=sprintf("%s 0", s); }
    else { s=sprintf("%s -1e+30", s); }
  }
  return s;
}
BEGIN{
  nsb = split(SBEG, SBEG_A);
  nse = split(SEND, SEND_A);
  for (i = 1; i <= nsb; ++i) {
    if (SBEG_A[i] !~ /^[0-9]+$/) {
      print "ERROR: Not an integer symbol \""SBEG_A[i]"\"" > "/dev/stderr";
      exit(1);
    }
    j = int(SBEG_A[i]);
    if (j < 0 || j >= DIM) {
      print "ERROR: Out-of-bounds index (idx: "j", dim: "DIM")" > "/dev/stderr";
      exit(1);
    }
  }
  for (i = 1; i <= nse; ++i) {
    if (SEND_A[i] !~ /^[0-9]+$/) {
      print "ERROR: Not an integer symbol \""SEND_A[i]"\"" > "/dev/stderr";
      exit(1);
    }
    j = int(SEND_A[i]);
    if (j < 0 || j >= DIM) {
      print "ERROR: Out-of-bounds index (idx: "j", dim: "DIM")" > "/dev/stderr";
      exit(1);
    }
  }
}{
  if ($2 == "[") {
    print;
    for (j = 1; j <= nsb; ++j) {
      print create_frame(DIM, SBEG_A[j]);
    }
  } else if ($NF == "]") {
    $NF="";
    print;
    for (j = 1; j <= nse; ++j) {
      print create_frame(DIM, SEND_A[j]);
    }
    print "]";
  } else {
    print;
  }

}' |
copy-matrix --print-args=false ark:- "$5";
