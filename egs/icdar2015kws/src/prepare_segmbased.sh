#!/bin/bash
set -e;

# Directory where the script is located.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "${SDIR}/..";


sed -r 's| ([^ ]+) ('\'') ([^ ]) | \1\2\3 |g' data/lang/lines/word/tr.txt;
