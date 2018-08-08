#!/usr/bin/env bash
set -e;
# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
# Move to the "root" of the experiment.
cd "$SDIR/..";
# Load useful functions
source "$PWD/../utils/functions_check.inc.sh" || exit 1;
export PATH="$PWD/../utils:$PATH";

mkdir -p data/corpus;

# download corpus
[ -s data/CScorpus_DB.tgz -o -s data/corpus/README ] ||
wget -P data/ --no-check-certificate \
  https://www.prhlt.upv.es/projects/multimodal/cs/files/CScorpus_DB.tgz || exit 1;

# extract it
[ -s data/corpus/README ] || tar -xzf data/CScorpus_DB.tgz -C data/corpus || exit 1;
rm -f data/CScorpus_DB.tgz;