#!/bin/bash
set -e;

base_url=https://download.pytorch.org/whl/cpu;

if [[ "$TRAVIS_PYTHON_VERSION" == 2.7 ]]; then
  url="${base_url}/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl";
elif [[ "$TRAVIS_PYTHON_VERSION" == 3.5 ]]; then
  url="${base_url}/torch-0.4.1-cp35-cp35m-linux_x86_64.whl";
elif [[ "$TRAVIS_PYTHON_VERSION" == 3.6 ]]; then
  url="${base_url}/torch-0.4.1-cp36-cp36m-linux_x86_64.whl";
elif [[ "$TRAVIS_PYTHON_VERSION" == 3.7 ]]; then
  url="${base_url}/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl";
else
  echo "Unsupported Python version: ${TRAVIS_PYTHON_VERSION}" >&2 && exit 1;
fi;

python -m pip install --progress-bar off "$url" || exit 1;
