#!/bin/sh
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
mkdir eigen_build
cd eigen-3.4.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="../../eigen_build" ../
make install 


