#!/bin/bash

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

echo "Checking dependencies..."

if nvcc --version; then
    printf "\033[32;1m CUDA compiler detected \033[0m\n"
else
    printf "\033[32;1mError:\033[0m you need to install nvcc in order to compile Silk\n"
    exit 1
fi

if cmake --version; then
    printf "\033[32;1m cmake detected \033[0m\n"
else
    printf "\033[32;1mError:\033[0m you need to install cmake in order to compile Silk\n"
    exit 1
fi

if git --version; then
    printf "\033[32;1m git detected \033[0m\n"
else
    printf "\033[32;1mError:\033[0m you need to install git in order to compile Silk\n"
    exit 1
fi

mkdir build

set -e

git submodule update --init --recursive

cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8

cd ..

cp "nnue/nnue_399M.dat" "build/src/silk_nnue.dat"

# build/src/silk
# build/test/gpu/gpu_unit_test
