#!/usr/bin/env bash
set -e;

if [ $# -ne 3 ]; then
    echo "Usage: ${0##*/} <imgdir> <gt-file> <outdir>" >&2;
    exit 1;
fi;

mkdir -p "$3";
split -d -a4 -l100 --additional-suffix=.txt "$2" "$3/";

np=0;
for f in "$3"/????.txt; do
  pageimg=${f/.txt/.png};
  [ -s "$pageimg" ] || {
    readarray -t arr < <(cut -d\  -f1 "$f" | xargs -n1 -I{} echo "$1/{}.png");
    convert -append "${arr[@]}" "$pageimg"  &
    ((++np));
    if [ $np -eq $(nproc) ]; then wait; np=0; fi;
  }
  ((++np));
done;
wait;


np=0;
for f in "$3"/????.txt; do
  pageimg=${f/.txt/.png};
  pagexml=${f/.txt/.xml};
  [ -s "$pagexml" ] || {
    WH=( $(identify -format '%w %h' "$pageimg") );

    awk -v DT="$(date +%Y-%m-%dT%X)" -v W=${WH[0]} -v H=${WH[1]} \
        -v PAGEIMG="$(basename "$pageimg")" -v IMGDIR="$1" \
    'BEGIN{
      print "<?xml version=\"1.0\" encoding=\"utf-8\"?>";
      print "<PcGts xmlns=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15\">";
      print "<Metadata>";
      print "<Creator>Joan Puigcerver</Creator>";
      print "<Created>" DT "</Created>";
      print "<LastChange>" DT "</LastChange>";
      print "</Metadata>";
      printf("<Page imageFilename=\"%s\" imageHeight=\"%d\" imageWidth=\"%d\">\n",
             PAGEIMG, H, W);
      printf("<TextRegion id=\"r1\" type=\"paragraph\">\n");
      printf("<Coords points=\"0,0 0,%d %d,%d %d,0\"/>\n", H, W, H, W);

      top=0;
    }{
      cmd = sprintf("identify -format \"%%w %%h\" %s", IMGDIR "/" $1 ".png");
      cmd | getline _; close(cmd);
      split(_, WH, " ");

      x0 = 0;
      x1 = WH[1];
      y0 = top;
      y1 = top + WH[2];

      printf("  <TextLine id=\"l%03d\">\n", NR);
      printf("  <Coords points=\"%d,%d %d,%d %d,%d %d,%d\"/>\n",
             x0, y0, x0, y1, x1, y1, x1, y0);
      printf("  <TextEquiv><Unicode>");
      for (i=2; i<=NF; ++i) {
        if ($i == "<sp>") $i=" ";
        printf("%s", $i);
      }
      printf("</Unicode></TextEquiv>\n");
      printf("  </TextLine>\n");

      top += WH[2];
    }END{
      print "</TextRegion>";
      print "</Page>";
      print "</PcGts>";
    }' "$f" > "$pagexml" &
    ((++np));
    if [ $np -eq $(nproc) ]; then wait; np=0; fi;
  }
  ((++n));
done;
wait;
