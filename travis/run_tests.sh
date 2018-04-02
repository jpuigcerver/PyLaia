#!/bin/bash
set -e;

find laia -name "*_test.py" |
sed 's|.py$||g;' | tr / . |
xargs -n1 python -m unittest;
