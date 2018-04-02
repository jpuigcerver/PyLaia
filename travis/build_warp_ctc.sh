#!/bin/bash
set -e;

# Build Baidu Warp-CTC
cd third_party/warp_ctc;
mkdir -p build;
cd build;
cmake ..;
make;

# Build and install PyTorch Wrapper
cd ../pytorch_binding;
python setup.py build;
python setup.py install;
