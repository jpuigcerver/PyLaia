#!/bin/bash
set -e;

# Build nnutils
cd third_party/nnutils;
mkdir -p build;
cd build;
cmake -DWITH_CUDA=OFF -DWITH_PYTORCH=ON ..;
make VERBOSE=1;

# Install PyTorch wrapper
cd pytorch;
python setup.py bdist_wheel;
pip install $(find dist/ -name "*.whl");
