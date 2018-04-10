#!/bin/bash
set -e;

VERSION=0.1.0+fade816;
BASE_URL=https://www.prhlt.upv.es/~jpuigcerver/nnutils/whl/cpu;

if [ "$TRAVIS_OS_NAME" = linux ]; then
    if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
	ARCH=cp27-cp27mu-linux_x86_64;
    elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
	ARCH=cp35-cp35m-linux_x86_64;
    elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
	ARCH=cp36-cp36m-linux_x86_64;
    fi;
fi;

URL="$BASE_URL/nnutils_pytorch-$VERSION-$ARCH.whl";
pip install "$URL";
