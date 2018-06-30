#!/bin/bash
set -e;
export LC_NUMERIC=C;

output="";
width=0.02
while [ "${1:0:1}" = "-" ]; do
  case "$1" in
    -o)
      output="$2"; shift 2;
      ;;
    -w)
      width="$2"; shift 2;
      ;;
    *)
      echo "Unknown option: \"$1\"" >&2 && exit 1;
  esac;
done;
[ $# -ne 1 ] && echo "Usage: ${0##*/} [-w binwidth] [-o output_tex] <file>" >&2 && exit 1;

N=$(wc -l "$1" | cut -d\  -f1);

if [ -z "$output" ]; then
  cat <<EOF | gnuplot -p
set boxwidth 0.9 relative
set style fill solid 1.0
set ylabel '% Pairs'
set xlabel 'P(R = 1 | x, y) - P_{theta}(R = 1 | x, y)'
bin(x)=$width*floor(x/$width)
plot '$1' using (bin(\$1)):(100.0 / $N) smooth freq with boxes notitle
EOF
else
  tmpdir="$(mktemp -d)";
  cat <<EOF | gnuplot
set terminal epslatex standalone
set output '$tmpdir/tmp.tex'
set boxwidth 0.9 relative
set style fill solid 1.0
set ylabel '\\% Pairs'
set xlabel '\$P(R = 1 \mid x, y) - P_{\theta}(R = 1 \mid x, y)\$'
bin(x)=$width*floor(x/$width)
plot '$1' using (bin(\$1)):(100.0 / $N) smooth freq with boxes notitle
EOF
  cd "$tmpdir";
  ps2pdf -dEPSCrop tmp-inc.eps tmp-inc.pdf
  pdflatex "tmp";
  echo $tmpdir;
fi;
