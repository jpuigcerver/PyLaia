#!/bin/bash
set -e;

if [ $# -ne 2 ]; then
  cat <<EOF > /dev/stderr
Usage: ${0##*/} PARTITION_ID CHECKPOINT OUTPUT_DIR

Example: ${0##*/} train/almazan/ctc/model.ckpt fsts/almazan/ctc
EOF
  exit 1;
fi;

export PYTHONPATH=$PWD/../..:$PYTHONPATH;

model="$1";
outdir="$2";

[ -d "$outdir" ] || mkdir -p "$outdir";

for p in va te; do
    [ -s "$outdir/$p.lat.ark" -a -s "$outdir/$p.lat.scp" ] || {
        python src/python/generate_ctc_lattice.py --add_softmax \
	        data/almazan/lang/syms_ctc.txt \
	        data/original/words \
	        data/almazan/lang/char/$p.txt \
	        "$model" \
	        >(lattice-remove-ctc-blank \
	            1 ark:- "ark,scp:$outdir/$p.lat.ark,$outdir/$p.lat.scp");
    }

    # Get 1-best path
    fn="$outdir/${p}_b0.fst";
    [ -s "$fn.ark" -a -s "$fn.scp" ] || {
        lattice-1best "ark:$outdir/$p.lat.ark" ark:- |
        lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
		    ark:- "ark,scp:$fn.ark,$fn.scp";
    }

    # Prune for different beam thresholds
    for beam in $(seq 20); do
        fn="$outdir/${p}_b${beam}.fst";
        [ -s "$fn.ark" -a -s "$fn.scp" ] || {
            lattice-prune --beam=$beam "ark:$outdir/$p.lat.ark" ark:- |
            lattice-to-fst --acoustic-scale=1 --lm-scale=1 \
		        ark:- "ark,scp:$fn.ark,$fn.scp";
        }
    done;
done;
