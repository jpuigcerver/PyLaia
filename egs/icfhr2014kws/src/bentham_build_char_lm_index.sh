#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd $SDIR/..;
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";
export PYTHONPATH="$PWD/../..:$PYTHONPATH";

beam=25;
lattice_beam=15;
ngram_order=7;
offset=-10;
prior_scale=0.3;
query_ngram_order=7;
query_nbest=100;
scale=8;
acoustic_scale=1.1;
graph_scale=1.0;
manual_scale_width=1.2;
manual_scale_height=2.0;
help_message="
Usage: ${0##*/} [options]

Options:
  --ngram_order  : (type = int, default = $ngram_order)
  --prior_scale  : (type = float, default = $prior_scale)
";
source "$PWD/../utils/parse_options.inc.sh" || exit 1;

base_dir="data/bentham/decode/char_${ngram_order}gram";
lats_dir="$base_dir/lats/ps${prior_scale}_b${beam}_lb${lattice_beam}";
query_dir="data/bentham/decode/query_char_${query_ngram_order}gram";
indx_dir="$base_dir/indx/ps${prior_scale}_b${beam}_lb${lattice_beam}/as${acoustic_scale}_gs${graph_scale}";
subs_dir="${indx_dir}/submissions";
check_all_files "$lats_dir/te.lat.ark" \
                "$lats_dir/va.lat.ark" \
                "$base_dir/chars.txt" \
                "$query_dir/queries.${query_nbest}best.txt" || exit 1;

wspace="$(grep "<sp>" "$base_dir/chars.txt" | awk '{print $2}' | tr \\n \ )";
marks="$(egrep "^[!?] " "$base_dir/chars.txt" | awk '{print $2}' | tr \\n \ )";
paren="$(egrep "^(\(|\)|\[|\]) " "$base_dir/chars.txt" | awk '{print $2}' | tr \\n \ )";

mkdir -p "$subs_dir";
for p in te va; do
  [ -s "$indx_dir/$p.pos.index" ] ||
  lattice-char-index-position \
    --nbest=10000 \
    --num-threads=$(nproc) \
    --acoustic-scale=${acoustic_scale} \
    --graph-scale=${graph_scale} \
    --other-groups="$marks ; $paren" "$wspace" \
    "ark:$lats_dir/$p.lat.ark" \
    "ark,t:$indx_dir/$p.pos.index";

  [ -s "$indx_dir/$p.seg.index" ] ||
  lattice-char-index-segment \
    --nbest=10000 \
    --num-threads=$(nproc) \
    --acoustic-scale=${acoustic_scale} \
    --graph-scale=${graph_scale} \
    --other-groups="$marks ; $paren" "$wspace" \
    "ark:$lats_dir/$p.lat.ark" \
    "ark,t:$indx_dir/$p.seg.index";

  [ -s "${subs_dir}/${p}_sw${manual_scale_width}_sh${manual_scale_height}.xml" ] ||
  extract_kws_index_bounding_box.py \
    --output_bounding_box \
    --resize_info_file=data/bentham/imgs/lines_h80/resize_info.txt \
    --symbols_table=${base_dir}/chars.txt \
    --global_scale=8 \
    --global_shift=-10 \
    position "${indx_dir}/$p.pos.index" \
    data/bentham/imgs/lines/fpgrams.txt \
    data/bentham/imgs/lines_h80 |
  ./src/build_qbe_xml.py \
    --scale_w="${manual_scale_width}" \
    --scale_h="${manual_scale_height}" \
    --biggest_regions=data/bentham/imgs/page_biggest_regions.txt \
    /dev/stdin \
    "${query_dir}/queries.${query_nbest}best.txt" \
    > "${subs_dir}/${p}_sw${manual_scale_width}_sh${manual_scale_height}.xml";
done;


[ -s "$indx_dir/auto_segmented_lines.pos.index" ] ||
lattice-char-index-position \
  --nbest=10000 \
  --num-threads=$(nproc) \
  --acoustic-scale=${acoustic_scale} \
  --graph-scale=${graph_scale} \
  --other-groups="$marks ; $paren" "$wspace" \
  "ark:$lats_dir/auto_segmented_lines.lat.ark" \
  "ark,t:$indx_dir/auto_segmented_lines.pos.index";

[ -s "$indx_dir/auto_segmented_lines.seg.index" ] ||
lattice-char-index-segment \
  --nbest=10000 \
  --num-threads=$(nproc) \
  --acoustic-scale=${acoustic_scale} \
  --graph-scale=${graph_scale} \
  --other-groups="$marks ; $paren" "$wspace" \
  "ark:$lats_dir/auto_segmented_lines.lat.ark" \
  "ark,t:$indx_dir/auto_segmented_lines.seg.index";

[ -s "${subs_dir}/auto_segmented_lines_sw1.2_sh2.0.xml" ] ||
extract_kws_index_bounding_box.py \
  --output_bounding_box \
  --resize_info_file=data/bentham/imgs/auto_segmented_lines_h80/resize_info.txt \
  --symbols_table=${base_dir}/chars.txt \
  --global_scale=8 \
  --global_shift=-10 \
  position "${indx_dir}/auto_segmented_lines.pos.index" \
  data/bentham/imgs/auto_segmented_lines/fpgrams.txt \
  data/bentham/imgs/auto_segmented_lines_h80 |
./src/build_qbe_xml.py \
  --scale_w=1.2 --scale_h=2.0 \
  /dev/stdin \
  "${query_dir}/queries.${query_nbest}best.txt" \
  > "${subs_dir}/auto_segmented_lines_sw1.2_sh2.0.xml";
