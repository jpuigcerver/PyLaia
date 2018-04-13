#!/bin/bash
set -e;

cv=cv4;

python steps/generate_ctc_lattice.py --add_softmax \
    train/dortmund/syms.txt \
    data/imgs/dortmund data/lang/dortmund/char/${cv}_te.txt \
    train/dortmund/ctc/lr0.0001_d/$cv/model.ckpt-valid-lowest-cer |
lattice-remove-ctc-blank 1 ark:- ark,scp:/data2/$cv.lat.ark,/data2/$cv.lat.scp;

for beam in $(seq 1 1 10); do
  fn=/data2/${cv}_b${beam}.fst
  [ -s $fn.ark -a -s $fn.scp ] || {
    join -1 1 <(sort queries_$cv.lst) <(sort /data2/$cv.lat.scp) |
    lattice-prune --beam=$beam scp:- ark:- |
    lattice-rmali ark:- ark:- |
    lattice-to-fst ark:- ark,scp:$fn.ark,$fn.scp;
  }
done;
