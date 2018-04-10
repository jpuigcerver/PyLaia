#!/bin/bash
set -e;
# NOTE: This requires that PyLaia is accessible from the PYTHONPATH.

# Move to PyLaia's root directory.
SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
cd "$SDIR/..";

# Find PyLaia tests modules.
readarray -t TEST_MODULES < <(find laia -name "*_test.py" |
				 sed 's|.py$||g;' | tr / .);

# Move to a different directory to execute the tests.
tmpdir=$(mktemp -d);
cd $tmpdir;
echo "${TEST_MODULES[@]}" | xargs -n1 python -m unittest;
rm -r $tmpdir;
