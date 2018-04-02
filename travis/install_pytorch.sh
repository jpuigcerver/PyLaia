#!/bin/bash
set -e;

VERSION=0.3.1;
BASE_URL=http://download.pytorch.org/whl;

if [ "$TRAVIS_OS_NAME" = linux ]; then
    if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
	ARCH=cp27-cp27mu-linux_x86_64;
    elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
	ARCH=cp35-cp35m-linux_x86_64;
    elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
	ARCH=cp36-cp36m-linux_x86_64;
    fi;
    BASE_URL="$BASE_URL/cpu";
elif [ "$TRAVIS_OS_NAME" = osx ]; then
    if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
	ARCH=cp27-none-macosx_10_6_x86_64;
    elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
	ARCH=cp35-cp35m-macosx_10_6_x86_64;
    elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
	ARCH=cp36-cp36m-macosx_10_7_x86_64;
    fi;
fi;

URL="$BASE_URL/torch-$VERSION-$ARCH.whl";
pip install "$URL";
