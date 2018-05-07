#!/usr/bin/env bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd $SDIR/..;

type htrsh_pageimg_forcealign &> /dev/null || {
    export PATH=$PWD/src/htrsh:$PATH;
    source $PWD/src/htrsh/htrsh.inc.sh;
}

exit 0;

outdir=data/uw3/pagexml_h44;
mkdir -p "$outdir";

if [[ ( ! -s "$outdir/000.png" ) || ( ! -s "$outdir/255.png" ) || \
      ( ! -s "$outdir/000.xml" ) || ( ! -s "$outdir/255.xml" ) ]]; then
    tmpdir=$(mktemp -d);
    $SDIR/create_pagexml.sh  \
        data/uw3/book_h44 data/uw3/lang/char/ascii/all.txt "$tmpdir";

    export htrsh_valschema=no;
    export htrsh_align_chars=yes;
    export htrsh_align_contour=no;
    export htrsh_align_isect=no;
    export htrsh_feat_deslope=no;
    export htrsh_feat_deslant=no;
    export htrsh_clean_type=none;

    np=0;
    for inpxml in "$tmpdir"/*.xml; do
        outxml="$outdir/$(basename $inpxml)";
        outlog="${outxml/.xml/.log}";
        [ -s $outxml ] && continue;
        echo "Aligning \"$inpxml\"..." >&2;
        ((++np));
        htrsh_pageimg_forcealign "$inpxml" "$outxml" -i 100 -e no &> "$outlog" &
        if [ "$np" -eq "$(nproc)" ]; then wait; np=0; fi;
    done;
    wait;

    # Move page PNG images.
    mv "$tmpdir"/???.png "$outdir";
    mv "$tmpdir"/???.txt "$outdir";
fi;

if [[ ( ! -s "data/uw3/book_h44_w20/000.png" ) || \
      ( ! -s "data/uw3/book_h44_w20/255.png" ) ]]; then
    for inpxml in "$outdir"/*.xml; do
        inptxt=${inpxml/.xml/.txt};
        xmlstarlet sel -t \
            -m '//_:Page/_:TextRegion/_:TextLine/_:Word/_:Glyph' \
            -v '@id' -o ' ' \
            -v '_:TextEquiv/_:Unicode' -o ' ' \
            -v '_:Coords/@points' -n "$inpxml" |
        awk -v PN=$(basename $inpxml .xml) 'BEGIN{
          line=""; word="";
          px = py = pw = ph = 0;
        }{
          split($1, I, "_");
          split($3, x0y1, ",");
          split($4, x0y0, ",");
          split($5, x1y0, ",");
          split($6, x1y0, ",");
          x = x0y0[1]; y = x0y0[2];
          w = x1y0[1] - x; h = x0y1[2] - y;
          if (line != "" && I[1] != line) {
            printf("\n");
          }
          if (line != I[1]) {
            printf("%s %d\t", PN, y);
          }
          #
          if (line == I[1] && word != I[2]) {
            printf(" <sp> %d", x - (px + pw));
          }
          printf(" %s %d", $2, w);
          line = I[1];
          word = I[2];
          px = x; py = y; pw = w; ph = h;
        }' | paste <(cut -d\  -f1 $inptxt) -;
    done | python src/create_fixed_size_dataset.py \
        --height=44 --width=20 "$outdir" data/uw3/book_h44_w20;
fi;
